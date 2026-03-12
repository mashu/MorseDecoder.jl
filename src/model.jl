
"""
    model.jl — Spectrogram encoder–decoder with cross-attention (Onion + Flux).

- Single-stream decoder: one output sequence per spectrogram, with speaker tokens
  (SPEAKER_1 … SPEAKER_6) so turn order is preserved. No fixed N slots; decode until EOS.
- Encoder: full self-attention over time. Decoder: causal self-attention + cross-attention to encoder.
- Loss: masked CE over sequence (PAD masked). Caller moves data to GPU.
"""

using Flux
using Onion
using Onion: TransformerBlock, RoPE, RMSNorm
using Einops: rearrange, @einops_str
using Random

# ─── Vocab (from vocab.jl) ───────────────────────────────────────────────────
# VOCAB_SIZE, START_TOKEN_IDX, PAD_TOKEN_IDX, EOS_TOKEN_IDX, SPEAKER_1_IDX..SPEAKER_6_IDX,
# TS_TOKEN_IDX, TE_TOKEN_IDX, speaker_token_id, is_speaker_token are defined in vocab.jl.
# Decoder logits size = VOCAB_SIZE (chars + <START> + PAD + <END> + [S1]..[S6] + [TS] + [TE]).

# ─── Encoder ─────────────────────────────────────────────────────────────────

"""
    SpectrogramEncoder(n_freq_bins, dim, n_heads, n_layers; ...)

Maps spectrogram to encoder hidden states. Conv1 → conv2 (kernel 3, pad 1) preserve time;
no downsampling so the transformer sees full spectrogram resolution for Morse dit/dah timing.

- Input: (freq_bins, batch, time) spectrogram.
- Output: (dim, num_tokens, batch) where num_tokens = time.
"""
struct SpectrogramEncoder
    conv1
    conv2
    blocks
    norm
    rope
end

Flux.@layer SpectrogramEncoder

function SpectrogramEncoder(
    n_freq_bins::Int,
    dim::Int,
    n_heads::Int,
    n_layers::Int;
    n_kv_heads::Int = n_heads,
    ff_mult::Int = 4,
    norm_eps::Float32 = 1f-5,
    max_len::Int = 4096,
)
    conv1 = Conv((3,), n_freq_bins => dim, gelu; pad=1)
    conv2 = Conv((3,), dim => dim, gelu; pad=1)
    blocks = [TransformerBlock(dim, n_heads, n_kv_heads, dim * ff_mult; norm_eps) for _ in 1:n_layers]
    norm = RMSNorm(dim; eps=norm_eps)
    rope = RoPE(dim ÷ n_heads, max_len)
    SpectrogramEncoder(conv1, conv2, blocks, norm, rope)
end

function (enc::SpectrogramEncoder)(spec::AbstractArray{T,3}) where T
    spec = log1p.(spec)
    x = permutedims(spec, (3, 1, 2))  # (time, freq_bins, batch)
    x = enc.conv1(x)                   # (time, dim, batch)
    x = enc.conv2(x)                   # (time, dim, batch)
    h = permutedims(x, (2, 1, 3))     # (dim, time, batch)
    n_tokens = size(h, 2)
    rope = enc.rope[1:n_tokens]
    for block in enc.blocks
        h = block(h; rope=rope, krope=rope)
    end
    enc.norm(h)
end

# ─── Decoder (causal self-attn + cross-attn) ──────────────────────────────────

"""
    SpectrogramDecoder(vocab_size, dim, n_heads, n_decoder_layers; n_cross_layers, decoder_input_dropout, ...)

Single-stream autoregressive decoder: causal self-attention then cross-attention to encoder.

**Layout (matches common audio-to-text practice, e.g. Whisper):**
- Encoder runs alone (self-attention only over audio); it does not use cross-attention.
- Decoder layers are **interleaved**: each of the first `n_decoder_layers` blocks is
  [decoder self-attn] → [cross-attn to encoder output]. So cross-attention is not
  independent; it is paired with a decoder self-attention layer.
- If n_cross_layers > n_decoder_layers, extra **cross-only** layers run after the
  interleaved stack (no extra decoder self-attn). This is an optional extension to
  increase encoder reliance.

- n_decoder_layers: number of decoder (self-attn) layers; each is followed by one cross-attn.
  Use 0 for a **cross-only** decoder (no self-attention; Whisper-Turbo style).
- n_cross_layers: total cross-attention layers (>= n_decoder_layers). When n_decoder_layers=0,
  all are cross-only; must be >= 1.
- decoder_input_dropout: dropout on decoder embeddings (after embed).
- self_attn_residual_scale: scale applied to the residual from each decoder self-attn block
  (default 1). Use < 1 (e.g. 0.5) to reduce reliance on decoder context and rely more on encoder.
"""
struct SpectrogramDecoder
    embed
    embed_dropout
    self_blocks
    cross_blocks
    norm
    head
    rope
    self_attn_residual_scale::Float32
end

Flux.@layer SpectrogramDecoder

function SpectrogramDecoder(
    vocab_size::Int,
    dim::Int,
    n_heads::Int,
    n_decoder_layers::Int;
    n_cross_layers::Int = n_decoder_layers,
    decoder_input_dropout::Real = 0.0f0,
    self_attn_residual_scale::Real = 1.0f0,
    n_kv_heads::Int = n_heads,
    ff_mult::Int = 4,
    norm_eps::Float32 = 1f-5,
    max_len::Int = 2048,
)
    n_cross_layers >= n_decoder_layers || throw(ArgumentError("n_cross_layers ($n_cross_layers) must be >= n_decoder_layers ($n_decoder_layers)"))
    (n_decoder_layers > 0 || n_cross_layers >= 1) || throw(ArgumentError("when n_decoder_layers=0, n_cross_layers must be >= 1"))
    embed = Flux.Embedding(vocab_size => dim)
    embed_dropout = Flux.Dropout(Float32(decoder_input_dropout))
    self_blocks = [TransformerBlock(dim, n_heads, n_kv_heads, dim * ff_mult; norm_eps) for _ in 1:n_decoder_layers]
    cross_blocks = [TransformerBlock(dim, n_heads, n_kv_heads, dim * ff_mult; norm_eps) for _ in 1:n_cross_layers]
    norm = RMSNorm(dim; eps=norm_eps)
    head = Dense(dim => vocab_size)
    rope = RoPE(dim ÷ n_heads, max_len)
    SpectrogramDecoder(embed, embed_dropout, self_blocks, cross_blocks, norm, head, rope, Float32(self_attn_residual_scale))
end

function (dec::SpectrogramDecoder)(decoder_input_ids::AbstractArray{<:Integer,2}, memory::AbstractArray{T,3}) where T
    h = dec.embed(decoder_input_ids)
    h = dec.embed_dropout(h)
    seq_len = size(h, 2)
    rope_dec = dec.rope[1:seq_len]
    # Self-attn: RoPE on q and k (decoder positions, same length).
    # Cross-attn: RoPE on q only. Use identity for krope so Zygote does not accum two
    # different-length RoPE gradients (dec 44 vs enc 269) in Onion's Attention pullback.
    # Encoder positions are already contextualized by the encoder's own RoPE.
    n_self = length(dec.self_blocks)
    scale = dec.self_attn_residual_scale
    for i in 1:n_self
        h_prev = h
        h = dec.self_blocks[i](h; rope=rope_dec, krope=rope_dec, causal=true)
        h = h_prev .+ scale .* (h .- h_prev)
        h = dec.cross_blocks[i](h, memory, memory; rope=rope_dec, krope=identity)
    end
    # Extra cross-only layers (n_cross_layers > n_decoder_layers)
    for i in (n_self + 1):length(dec.cross_blocks)
        h = dec.cross_blocks[i](h, memory, memory; rope=rope_dec, krope=identity)
    end
    dec.head(dec.norm(h))
end

# ─── Full model ───────────────────────────────────────────────────────────────

"""
    SpectrogramEncoderDecoder(encoder, decoder)

Full model: encoder maps spectrogram to memory; decoder maps (decoder_input_ids, memory) to logits.
"""
struct SpectrogramEncoderDecoder
    encoder::SpectrogramEncoder
    decoder::SpectrogramDecoder
end

Flux.@layer SpectrogramEncoderDecoder

function (m::SpectrogramEncoderDecoder)(spec, decoder_input_ids, memory=nothing)
    enc_mem = something(memory, m.encoder(spec))
    logits = m.decoder(decoder_input_ids, enc_mem)  # single stream: (vocab, seq, batch)
    return logits
end

# ─── Single-stream batching ───────────────────────────────────────────────────

"""
    prepare_decoder_batch(targets; start_token, pad_token)

From Batch targets (batch, max_seq) build decoder input and target for teacher forcing.
- `decoder_input`: (max_seq, batch), [START, t1, …, t_{L-1}].
- `decoder_target`: (max_seq, batch), [t1, …, t_L]. Padding (0) is mapped to pad_token.
"""
function prepare_decoder_batch(
    targets::AbstractArray{<:Integer,2};
    start_token::Int = START_TOKEN_IDX,
    pad_token::Int = PAD_TOKEN_IDX,
)
    B, L = size(targets)
    # targets (B, L) -> (L, B) for decoder
    target_T = permutedims(targets, (2, 1))   # (L, B)
    input_tail = target_T[1:max(1, L) - 1, :]  # (L-1, B)
    decoder_input = vcat(fill(start_token, 1, B), input_tail)
    decoder_target = target_T
    decoder_input = ifelse.(decoder_input .== 0, pad_token, decoder_input)
    decoder_target = ifelse.(decoder_target .== 0, pad_token, decoder_target)
    (; decoder_input, decoder_target)
end

# ─── Loss (masked cross-entropy over sequence) ─────────────────────────────────

"""
    sequence_cross_entropy(logits, decoder_target; pad_idx)

Logits (vocab, seq, batch), decoder_target (seq, batch). Mean CE over non-pad positions.
"""
function sequence_cross_entropy(
    logits::AbstractArray{T,3},
    decoder_target::AbstractArray{<:Integer,2};
    pad_idx::Int = PAD_TOKEN_IDX,
) where T
    vocab, seq_len, batch = size(logits)
    log_probs = Flux.logsoftmax(logits; dims=1)
    nll_flat = -sum(Flux.onehotbatch(vec(decoder_target), 1:vocab) .* reshape(log_probs, vocab, :); dims=1)
    nll = reshape(nll_flat, seq_len, batch)
    valid = decoder_target .!= pad_idx
    total_valid = max(sum(valid), 1)
    sum(nll .* valid) / total_valid
end

# ─── Training step (one batch) ───────────────────────────────────────────────

"""
    prepare_training_batch(batch) -> (spec, decoder_input, decoder_target)
"""
function prepare_training_batch(batch::Batch)
    dec = prepare_decoder_batch(batch.targets)
    (batch.spectrogram, dec.decoder_input, dec.decoder_target)
end

"""
    train_step(model, spec, decoder_input, decoder_target; encoder_dropout)

Single training step. 100% teacher forcing: decoder input is always ground truth.
"""
function train_step(
    model::SpectrogramEncoderDecoder,
    spec,
    decoder_input,
    decoder_target;
    encoder_dropout::Real = 0.0,
)
    memory = model.encoder(spec)
    if encoder_dropout > 0 && rand(Float32) < encoder_dropout
        memory = memory .* 0f0
    end
    logits = model.decoder(decoder_input, memory)
    sequence_cross_entropy(logits, decoder_target)
end

train_step(model::SpectrogramEncoderDecoder, batch::Batch; kws...) =
    train_step(model, prepare_training_batch(batch)...; kws...)

# ─── Autoregressive sampling ───────────────────────────────────────────────────

"""
    decode_autoregressive(model, spec; max_len, start_token, to_device, batch_size)

Single-stream decode: one output sequence per spectrogram (with speaker tokens). Stops at EOS or max_len.
Returns (seq_len, batch) token indices. to_device(x) places x on the same device as spec.
"""
function decode_autoregressive(
    model::SpectrogramEncoderDecoder,
    spec;
    max_len::Int = 256,
    start_token::Int = START_TOKEN_IDX,
    to_device = identity,
    batch_size::Int = size(spec, 2),
)
    memory = model.encoder(spec)
    ids_so_far = to_device(fill(start_token, 1, batch_size))

    for _ in 2:max_len
        logits = model.decoder(ids_so_far, memory)
        next_logits = selectdim(logits, 2, size(logits, 2))
        next_logits_cpu = Array(next_logits)
        next_logits_cpu[PAD_TOKEN_IDX, :] .= -1f10
        next_logits_cpu[START_TOKEN_IDX, :] .= -1f10
        am = argmax(next_logits_cpu; dims=1)
        next_ids_cpu = reshape(map(i -> i[1], am), 1, batch_size)
        next_ids = to_device(next_ids_cpu)
        ids_so_far = cat(ids_so_far, next_ids; dims=1)
        # Stop when all batch positions have emitted EOS
        all(i -> i == EOS_TOKEN_IDX, next_ids_cpu) && break
    end

    ids_so_far
end
