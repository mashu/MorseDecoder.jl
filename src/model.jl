
"""
    model.jl — Spectrogram encoder–decoder with cross-attention (Onion + Flux).

- Single-stream decoder: one output sequence per spectrogram, with speaker tokens
  (SPEAKER_1 … SPEAKER_6) so turn order is preserved. No fixed N slots; decode until EOS.
- Encoder: full self-attention over time. Decoder: causal self-attention + cross-attention to encoder.
- Optional CTC head on encoder output for streaming / joint CTC-attention training.
- Loss: masked CE over sequence (PAD masked) + optional CTC loss. Caller moves data to GPU.
- Balance: use a modest CTC weight (e.g. 0.15–0.2) so attention stays dominant; CTC mainly helps
  alignment and streaming. If CTC underperforms, try a larger encoder (--encoder-dim > dim or
  more --encoder-layers) so the encoder has enough capacity for frame-level predictions.
"""

using Flux
using KernelAbstractions: KernelAbstractions as KA, get_backend, synchronize, @kernel, @index, @uniform
using Onion
using Onion: TransformerBlock, RoPE, RMSNorm
using Einops: rearrange, @einops_str
using Random

# ─── Vocab (from vocab.jl) ───────────────────────────────────────────────────
# VOCAB_SIZE, START_TOKEN_IDX, PAD_TOKEN_IDX, EOS_TOKEN_IDX, SPEAKER_1_IDX..SPEAKER_6_IDX,
# TS_TOKEN_IDX, TE_TOKEN_IDX, speaker_token_id, is_speaker_token are defined in vocab.jl.
# Decoder logits size = VOCAB_SIZE (chars + <START> + PAD + <END> + [S1]..[S6] + [TS] + [TE]).

# CTC blank token is appended after VOCAB_SIZE (NNlib convention: blank = last class).
const CTC_VOCAB_SIZE = VOCAB_SIZE + 1
const CTC_BLANK_IDX  = CTC_VOCAB_SIZE

# ─── CW front-end (CWFeatureExtractor) ─────────────────────────────────────
#
# Full mel in; one stride-2 downsample → T/2 so transformer gets ~4 frames/dot at 50 WPM.
# Two convs: conv7 → gelu, conv4 stride-2 → gelu. No BatchNorm.
#
"""
    CWFeatureExtractor(n_freq_bins; d_model=128)

Minimal conv front-end: (T, n_freq_bins, B) → (T/2, d_model, B).
Stack: conv7 → gelu, conv4 stride2 → gelu.
"""
struct CWFeatureExtractor
    conv1::Conv
    conv2::Conv
end
Flux.@layer CWFeatureExtractor

function CWFeatureExtractor(n_freq_bins::Int; d_model::Int=128)
    CWFeatureExtractor(
        Conv((7,), n_freq_bins => 64; pad=3),
        Conv((4,), 64 => d_model; stride=2, pad=1),
    )
end

function (m::CWFeatureExtractor)(x)
    h = m.conv1(x) |> gelu   # (T, 64, B)
    h = m.conv2(h) |> gelu   # (T/2, d_model, B)
    h
end

# ─── Encoder ───────────────────────────────────────────────────────────────

"""Encoder time is spectrogram time ÷ this (CW front-end has one stride-2 → T/2)."""
const ENCODER_DOWNSAMPLE = 2

const CW_FRONTEND_DIM = 128

"""
    SpectrogramEncoder(n_freq_bins, dim, n_heads, n_layers; ...)

Maps spectrogram to encoder hidden states. **Frontend:** CWFeatureExtractor (conv7 → gelu, conv4 stride-2 → gelu; one downsampling → T/2). **Then:** transformer blocks.

**Time resolution:** Spectrogram ~344 Hz (hop 128 @ 44.1 kHz); after 2× downsampling the transformer
sees ~172 Hz. At 50 WPM a dot ≈ 24 ms → ~4 frames per dot, dash ~12 frames. Gives the transformer
enough resolution to be robust to noise and distinguish dot vs dash.

- Input: (freq_bins, batch, time) spectrogram in **log10 scale** (same as training:
  MorseSimulator peak-norm + log10). For real audio use `spectrogram_to_model_scale` first.
- Output: (dim, num_tokens, batch) where num_tokens = time ÷ 2.
"""
struct SpectrogramEncoder
    frontend::CWFeatureExtractor
    proj::Union{Dense,Nothing}  # 128 → dim when dim != 128
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
    norm_eps::AbstractFloat = 1f-5,
    max_len::Int = 4096,
)
    frontend = CWFeatureExtractor(n_freq_bins; d_model=CW_FRONTEND_DIM)
    proj = dim == CW_FRONTEND_DIM ? nothing : Dense(CW_FRONTEND_DIM => dim)
    blocks = [TransformerBlock(dim, n_heads, n_kv_heads, dim * ff_mult; norm_eps) for _ in 1:n_layers]
    norm = RMSNorm(dim; eps=norm_eps)
    rope = RoPE(dim ÷ n_heads, max_len)
    SpectrogramEncoder(frontend, proj, blocks, norm, rope)
end

function apply_encoder_proj(proj::Dense, h)
    T_enc, B = size(h, 2), size(h, 3)
    out = proj(reshape(h, size(h, 1), :))
    reshape(out, size(out, 1), T_enc, B)
end
apply_encoder_proj(::Nothing, h) = h

function (enc::SpectrogramEncoder)(spec::AbstractArray{T,3}) where T
    x = permutedims(spec, (3, 1, 2))   # (time, freq_bins, batch)
    h = enc.frontend(x)                 # (T/2, 128, batch)
    h = permutedims(h, (2, 1, 3))      # (128, T/2, batch)
    h = apply_encoder_proj(enc.proj, h)
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
    self_attn_residual_scale::AbstractFloat
end

Flux.@layer SpectrogramDecoder

function SpectrogramDecoder(
    vocab_size::Int,
    dim::Int,
    n_heads::Int,
    n_decoder_layers::Int;
    n_cross_layers::Int = n_decoder_layers,
    decoder_input_dropout::AbstractFloat = 0.0f0,
    self_attn_residual_scale::AbstractFloat = 1.0f0,
    n_kv_heads::Int = n_heads,
    ff_mult::Int = 4,
    norm_eps::AbstractFloat = 1f-5,
    max_len::Int = 2048,
)
    n_cross_layers >= n_decoder_layers || throw(ArgumentError("n_cross_layers ($n_cross_layers) must be >= n_decoder_layers ($n_decoder_layers)"))
    (n_decoder_layers > 0 || n_cross_layers >= 1) || throw(ArgumentError("when n_decoder_layers=0, n_cross_layers must be >= 1"))
    embed = Flux.Embedding(vocab_size => dim)
    embed_dropout = Flux.Dropout(decoder_input_dropout)
    self_blocks = [TransformerBlock(dim, n_heads, n_kv_heads, dim * ff_mult; norm_eps) for _ in 1:n_decoder_layers]
    cross_blocks = [TransformerBlock(dim, n_heads, n_kv_heads, dim * ff_mult; norm_eps) for _ in 1:n_cross_layers]
    norm = RMSNorm(dim; eps=norm_eps)
    head = Dense(dim => vocab_size)
    rope = RoPE(dim ÷ n_heads, max_len)
    SpectrogramDecoder(embed, embed_dropout, self_blocks, cross_blocks, norm, head, rope, self_attn_residual_scale)
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
    SpectrogramEncoderDecoder(encoder, decoder, ctc_head; encoder_proj=nothing)

Full model: encoder → two sibling heads (CTC and decoder), not in sequence.

  spec → Encoder → enc_mem
            ├→ ctc_head(enc_mem)     → CTC logits (per frame; blank, collapse repeats)
            └→ [encoder_proj] → dec_mem → Decoder(ids, dec_mem) → decoder logits (per step; START/EOS)

CTC and decoder both consume the same encoder output; they are trained jointly (CE + CTC loss)
but produce different hypotheses (frame-level vs autoregressive), so their outputs need not match.
When encoder_dim != decoder dim, `encoder_proj` maps enc_mem → dec_mem for the decoder;
CTC head always uses raw enc_mem.
"""
struct SpectrogramEncoderDecoder
    encoder::SpectrogramEncoder
    decoder::SpectrogramDecoder
    ctc_head::Dense   # encoder output → CTC logits (CTC_VOCAB_SIZE)
    encoder_proj::Union{Dense,Nothing}  # encoder_dim => decoder_dim when encoder_dim != decoder_dim
end

Flux.@layer SpectrogramEncoderDecoder

SpectrogramEncoderDecoder(encoder::SpectrogramEncoder, decoder::SpectrogramDecoder, ctc_head::Dense) =
    SpectrogramEncoderDecoder(encoder, decoder, ctc_head, nothing)
function SpectrogramEncoderDecoder(encoder::SpectrogramEncoder, decoder::SpectrogramDecoder)
    dim = size(decoder.embed.weight, 2)  # decoder hidden dim
    SpectrogramEncoderDecoder(encoder, decoder, Dense(dim => CTC_VOCAB_SIZE), nothing)
end

"""Dispatch: no projection (same dim) vs Dense projection."""
project_memory(::Nothing, x) = x
project_memory(p::Dense, x) = p(x)

"""Return (enc_mem, dec_mem): raw encoder output (for CTC) and memory for decoder (projected via encoder_proj)."""
function encode(m::SpectrogramEncoderDecoder, spec)
    enc_mem = m.encoder(spec)
    (enc_mem, project_memory(m.encoder_proj, enc_mem))
end

function (m::SpectrogramEncoderDecoder)(spec, decoder_input_ids)
    logits = m.decoder(decoder_input_ids, last(encode(m, spec)))
    return logits
end
function (m::SpectrogramEncoderDecoder)(spec, decoder_input_ids, memory)
    logits = m.decoder(decoder_input_ids, memory)
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
    sequence_cross_entropy(logits, decoder_target; pad_idx, label_smoothing)

Logits (vocab, seq, batch), decoder_target (seq, batch). Mean CE over non-pad positions.
When label_smoothing > 0, targets are smoothed toward uniform to reduce overconfident collapse.
"""
function sequence_cross_entropy(
    logits::AbstractArray{T,3},
    decoder_target::AbstractArray{<:Integer,2};
    pad_idx::Int = PAD_TOKEN_IDX,
    label_smoothing::T = zero(T),
) where T
    vocab, seq_len, batch = size(logits)
    log_probs = Flux.logsoftmax(logits; dims=1)
    nll_flat = -sum(Flux.onehotbatch(vec(decoder_target), 1:vocab) .* reshape(log_probs, vocab, :); dims=1)
    nll = reshape(nll_flat, seq_len, batch)
    if label_smoothing > 0
        ε = label_smoothing
        mean_log_prob = reshape(sum(log_probs; dims=1) / vocab, seq_len, batch)
        nll = (1 - ε) .* nll .+ ε .* (-mean_log_prob)
    end
    valid = decoder_target .!= pad_idx
    total_valid = max(sum(valid), 1)
    sum(nll .* valid) / total_valid
end

# ─── CTC target preparation ──────────────────────────────────────────────────

"""
    prepare_ctc_targets(batch::Batch) -> Vector{Vector{Int}}

Extract CTC label sequences from a Batch: **one target sequence per sample** (from that sample's
transcript in batch.targets). Strips START, PAD, EOS; keeps chars, speaker tokens, [TS], [TE].
Truncates so encoder length (spectrogram ÷ ENCODER_DOWNSAMPLE) >= 2*L+1 for CTC.
Targets vary per sample and per batch (each batch has different random transcripts from MorseSimulator).
"""
function prepare_ctc_targets(batch::Batch)
    prepare_ctc_targets(batch, div.(batch.input_lengths, ENCODER_DOWNSAMPLE))
end

function prepare_ctc_targets(batch::Batch, enc_lengths::Vector{Int})
    B = size(batch.targets, 1)
    ctc_targets = Vector{Vector{Int}}(undef, B)
    for b in 1:B
        tgt = @view batch.targets[b, 1:batch.target_lengths[b]]
        raw = [t for t in tgt if t != START_TOKEN_IDX && t != PAD_TOKEN_IDX && t != EOS_TOKEN_IDX && t != 0]
        T_b = enc_lengths[b]
        L_max = max(0, div(T_b - 1, 2))  # CTC needs T >= 2*L+1
        ctc_targets[b] = length(raw) <= L_max ? raw : raw[1:L_max]
    end
    ctc_targets
end

# ─── Training step (one batch) ───────────────────────────────────────────────

"""
    prepare_training_batch(batch) -> (spec, decoder_input, decoder_target)
"""
function prepare_training_batch(batch::Batch)
    dec = prepare_decoder_batch(batch.targets)
    (batch.spectrogram, dec.decoder_input, dec.decoder_target)
end

"""Dispatch: no CTC targets → return loss unchanged."""
add_ctc_loss(loss, _model, _enc_mem, ::Nothing, _input_lengths, _ctc_weight) = loss
function add_ctc_loss(loss, model::SpectrogramEncoderDecoder, enc_mem, ctc_targets::Vector{Vector{Int}}, input_lengths::Vector{Int}, ctc_weight::Real)
    # input_lengths = encoder frame lengths: div.(batch.input_lengths, ENCODER_DOWNSAMPLE)
    loss + ctc_weight * CTCLoss.ctc_loss_batched(model.ctc_head(enc_mem), ctc_targets, input_lengths, CTC_BLANK_IDX)
end

"""
    train_step(model, spec, decoder_input, decoder_target; ctc_targets, input_lengths, ctc_weight)

Single training step. 100% teacher forcing for decoder. When `ctc_targets` is provided,
adds CTC loss: `loss = CE + ctc_weight * CTC`. Pass `ctc_targets=nothing` to skip CTC.
`ctc_targets` is Vector{Vector{Int}} (CPU), `input_lengths` is Vector{Int} (CPU).
"""
function train_step(
    model::SpectrogramEncoderDecoder,
    spec,
    decoder_input,
    decoder_target;
    ctc_targets = nothing,
    input_lengths = nothing,
    ctc_weight::AbstractFloat = 0.0f0,
    label_smoothing::AbstractFloat = 0.0f0,
)
    enc_mem, dec_mem = encode(model, spec)
    logits = model.decoder(decoder_input, dec_mem)
    loss = sequence_cross_entropy(logits, decoder_target; label_smoothing)
    add_ctc_loss(loss, model, enc_mem, ctc_targets, input_lengths, ctc_weight)
end

train_step(model::SpectrogramEncoderDecoder, batch::Batch; kws...) =
    train_step(model, prepare_training_batch(batch)...; kws...)

# ─── Autoregressive sampling ───────────────────────────────────────────────────

"""Shared loop: run decoder from (memory, ids_buf, len_so_far) until max_len or EOS. Returns ids_buf[1:len_so_far, :]."""
function autoregressive_loop(
    model::SpectrogramEncoderDecoder,
    memory,
    ids_buf,
    len_so_far::Int;
    max_len::Int,
    to_device = identity,
)
    batch_size = size(ids_buf, 2)
    for _ in (len_so_far + 1):max_len
        ids_so_far = copy(ids_buf[1:len_so_far, :])
        logits = model.decoder(ids_so_far, memory)
        next_logits = selectdim(logits, 2, size(logits, 2))
        next_logits[PAD_TOKEN_IDX, :] .= -1f10
        next_logits[START_TOKEN_IDX, :] .= -1f10
        next_ids = argmax_dim1(next_logits)
        len_so_far += 1
        ids_buf[len_so_far, :] .= vec(next_ids)
        all(==(EOS_TOKEN_IDX), next_ids) && break
    end
    ids_buf[1:len_so_far, :]
end

"""
    decode_autoregressive(model, spec; max_len, start_token, to_device, batch_size)

Single-stream decode: one output sequence per spectrogram (with speaker tokens). Stops at EOS or max_len.
Returns (seq_len, batch) token indices.
"""
function decode_autoregressive(
    model::SpectrogramEncoderDecoder,
    spec;
    max_len::Int = 256,
    start_token::Int = START_TOKEN_IDX,
    to_device = identity,
    batch_size::Int = size(spec, 2),
)
    _, memory = encode(model, spec)
    ids_buf = to_device(fill(start_token, max_len, batch_size))
    autoregressive_loop(model, memory, ids_buf, 1; max_len, to_device)
end

"""
    decode_autoregressive(model, spec, initial_tokens; max_len, to_device, batch_size)

Decode continuing from prefix `initial_tokens` (for chunk-by-chunk conversation decoding).
"""
function decode_autoregressive(
    model::SpectrogramEncoderDecoder,
    spec,
    initial_tokens::AbstractVector{<:Integer};
    max_len::Int = 256,
    to_device = identity,
    batch_size::Int = 1,
)
    _, memory = encode(model, spec)
    P = length(initial_tokens)
    P >= max_len && return to_device(reshape(collect(initial_tokens), P, 1))[1:max_len, :]
    ids_buf = to_device(fill(START_TOKEN_IDX, max_len, batch_size))
    ids_buf[1:P, 1] .= vec(to_device(collect(Int, initial_tokens)))
    autoregressive_loop(model, memory, ids_buf, P; max_len, to_device)
end

"""
    decode_conversation(model, chunks, to_device; max_len_per_chunk, start_token)

Decode a full conversation from an iterable of spectrogram chunks. Runs autoregressive
decode chunk by chunk; each chunk continues from the previous output (using [TS]/[TE]
and EOS). Stops when EOS is emitted or chunks are exhausted. Returns a single token-id vector.

**Chunk source is abstract:** `chunks` can be from the simulator (`ChunkedConversation`),
from live audio (mic/radio), or from files. Each element must be a spectrogram array:
either (n_bins, time) or (n_bins, 1, time), with the same n_bins and scale as training.
For real audio use `compute_spectrogram` then `spectrogram_to_model_scale` before
slicing into chunks so the scale matches the encoder.
"""
function decode_conversation(
    model::SpectrogramEncoderDecoder,
    chunks,
    to_device = identity;
    max_len_per_chunk::Int = 256,
    start_token::Int = START_TOKEN_IDX,
)
    tokens = [start_token]
    for chunk in chunks
        spec_3d = to_device(spec_for_decode(chunk))
        out = decode_autoregressive(
            model, spec_3d, tokens;
            max_len = max_len_per_chunk,
            to_device = to_device,
            batch_size = 1,
        )
        tokens = vec(collect(out))
        if !isempty(tokens) && tokens[end] == EOS_TOKEN_IDX
            break
        end
    end
    tokens
end

# Dispatch: (n_bins, time) → (n_bins, 1, time); already 3D passed through.
spec_for_decode(spec::AbstractMatrix) = reshape(spec, size(spec, 1), 1, size(spec, 2))
spec_for_decode(spec::AbstractArray{<:Any,3}) = spec

# Accept ChunkedConversation (yields Sample); forward to spectrogram iterable.
decode_conversation(model::SpectrogramEncoderDecoder, chunks::ChunkedConversation, to_device = identity; kws...) =
    decode_conversation(model, (s.spectrogram for s in chunks), to_device; kws...)

# Argmax along dim 1 (vocab) via KernelAbstractions; avoids CuArray's tuple reduction (isless/convert) that fails on some GPUs.
@kernel function argmax_dim1_kernel(logits, out)
    b = @index(Global)
    @uniform vocab_size = size(logits, 1)
    n_batch = size(logits, 2)
    if b <= n_batch
        best_i = 1
        best_v = @inbounds logits[1, b]
        for i in 2:vocab_size
            v = @inbounds logits[i, b]
            if v > best_v
                best_v = v
                best_i = i
            end
        end
        @inbounds out[b] = best_i
    end
end

function argmax_dim1(logits)
    backend = get_backend(logits)
    if backend isa KA.GPU
        batch_size = size(logits, 2)
        out = similar(logits, Int32, (batch_size,))
        argmax_dim1_kernel(backend, 256)(logits, out; ndrange=batch_size)
        synchronize(backend)
        reshape(Int.(out), 1, batch_size)
    else
        am = argmax(logits; dims=1)
        reshape((x -> x[1]).(am), 1, size(logits, 2))
    end
end

# ─── CTC greedy decode ───────────────────────────────────────────────────────

"""
    ctc_greedy_decode(model, spec; input_lengths) -> Vector{Vector{Int}}

Run encoder + CTC head, then greedy-decode: argmax per frame, collapse consecutive
duplicates, remove blanks. Returns one Vector{Int} of token IDs per batch element.
"""
function ctc_greedy_decode(
    model::SpectrogramEncoderDecoder,
    spec;
    input_lengths::Vector{Int} = fill(size(spec, 3), size(spec, 2)),
)
    enc_mem, _ = encode(model, spec)  # CTC uses raw encoder output
    ctc_logits = model.ctc_head(enc_mem)  # (CTC_VOCAB_SIZE, time, batch); keep on same device as enc_mem
    enc_lengths = div.(input_lengths, ENCODER_DOWNSAMPLE)
    CTCLoss.ctc_greedy_decode(ctc_logits, enc_lengths; blank = CTC_BLANK_IDX)
end

"""
    ctc_greedy_decode(ctc_logits; input_lengths) -> Vector{Vector{Int}}

Greedy CTC decode from raw logits (CTC_VOCAB_SIZE, time, batch).
"""
function ctc_greedy_decode(
    ctc_logits::AbstractArray{<:Real,3};
    input_lengths::Vector{Int} = fill(size(ctc_logits, 2), size(ctc_logits, 3)),
)
    CTCLoss.ctc_greedy_decode(ctc_logits, input_lengths; blank = CTC_BLANK_IDX)
end
