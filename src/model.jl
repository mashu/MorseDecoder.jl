"""
    model.jl — Spectrogram encoder–decoder with cross-attention (Onion + Flux).

- Spectrogram is chunked into non-overlapping time intervals as tokens.
- Encoder processes tokens (full self-attention); Decoder is autoregressive
  (causal self-attention + cross-attention to encoder).
- Multi-station: one decoder, batch = batch_size × max_stations (encoder output
  repeated per station); one output head. Loss is masked over valid (batch, station).
- No mutations in forward pass; Zygote- and CUDA-friendly (no indexing in layers).
- Caller moves data to GPU; no device branching in the model.
"""

using Flux
using Onion
using Onion: TransformerBlock, RoPE, RMSNorm
using Einops: rearrange, @einops_str
using Random

# ─── Vocab and constants ─────────────────────────────────────────────────────

"""Vocabulary size: chars + start + pad + end-of-sequence."""
const VOCAB_SIZE = NUM_CHARS + 3
const START_TOKEN_IDX = NUM_CHARS + 1
const PAD_TOKEN_IDX = NUM_CHARS + 2   # padding in batch; mask in loss
# EOS = "no more content for this slot (in this chunk)" — e.g. silence, end of exchange, or end of buffer.
# Not "station left the band": one calling station can have many short exchanges (73 with A, then B, …);
# each chunk may show one exchange; EOS stops decoding until the next chunk has new content.
const EOS_TOKEN_IDX = NUM_CHARS + 3

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
    SpectrogramDecoder(vocab_size, dim, n_heads, n_layers; ...)

Autoregressive decoder: causal self-attention then cross-attention to encoder memory.
- `embed`: token embedding
- `station_embed`: per-station embedding so different stations can produce different outputs (same encoder view).
- `self_blocks`: causal self-attention
- `cross_blocks`: cross-attention (q = decoder, k,v = encoder memory)
- `norm`, `head`: output logits
"""
struct SpectrogramDecoder
    embed
    station_embed
    self_blocks
    cross_blocks
    norm
    head
    rope
end

Flux.@layer SpectrogramDecoder

function SpectrogramDecoder(
    vocab_size::Int,
    dim::Int,
    n_heads::Int,
    n_layers::Int;
    n_kv_heads::Int = n_heads,
    ff_mult::Int = 4,
    norm_eps::Float32 = 1f-5,
    max_len::Int = 2048,
    max_stations::Int = 5,
)
    embed = Flux.Embedding(vocab_size => dim)
    station_embed = Flux.Embedding(max_stations => dim)
    self_blocks = [TransformerBlock(dim, n_heads, n_kv_heads, dim * ff_mult; norm_eps) for _ in 1:n_layers]
    cross_blocks = [TransformerBlock(dim, n_heads, n_kv_heads, dim * ff_mult; norm_eps) for _ in 1:n_layers]
    norm = RMSNorm(dim; eps=norm_eps)
    head = Dense(dim => vocab_size)
    rope = RoPE(dim ÷ n_heads, max_len)
    SpectrogramDecoder(embed, station_embed, self_blocks, cross_blocks, norm, head, rope)
end

function (dec::SpectrogramDecoder)(decoder_input_ids::AbstractArray{<:Integer,2}, memory::AbstractArray{T,3}, station_ids::AbstractArray{<:Integer,2}) where T
    h = dec.embed(decoder_input_ids) .+ dec.station_embed(station_ids)
    seq_len = size(h, 2)
    rope_dec = dec.rope[1:seq_len]
    # Self-attn: RoPE on q and k (decoder positions, same length).
    # Cross-attn: RoPE on q only. Use identity for krope so Zygote does not accum two
    # different-length RoPE gradients (dec 44 vs enc 269) in Onion's Attention pullback.
    # Encoder positions are already contextualized by the encoder's own RoPE.
    for (sblock, cblock) in zip(dec.self_blocks, dec.cross_blocks)
        h = sblock(h; rope=rope_dec, krope=rope_dec, causal=true)
        h = cblock(h, memory, memory; rope=rope_dec, krope=identity)
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
    logits = m.decoder(decoder_input_ids, enc_mem)
    return logits
end

# ─── Multi-station batching ──────────────────────────────────────────────────

"""
    prepare_decoder_batch(targets, n_stations; start_token)

From Batch targets (batch, max_stations, max_seq) build decoder input and flat targets
for batch = batch_size × max_stations. Broadcast/reshape only; no loops or indexing.

- `decoder_input`: (max_seq, batch × max_stations), [START, t1, …, t_{L-1}] per column (teacher forcing).
- `decoder_target`: (max_seq, batch × max_stations), [t1, …, t_L]; loss at position i = CE(logits[i], t_i).
- `station_mask`: (batch × max_stations,) true where (b,k) has k ≤ n_stations[b].
  Collate uses 0 for padding; we map 0 → PAD_TOKEN_IDX so embedding and loss use a valid token.
"""
function prepare_decoder_batch(
    targets::AbstractArray{<:Integer,3},
    n_stations::AbstractVector{<:Integer};
    start_token::Int = START_TOKEN_IDX,
    pad_token::Int = PAD_TOKEN_IDX,
)
    B, K, L = size(targets)
    # (L-1, B, K) -> (L-1, B*K); when L==1 this is (0, B*K)
    input_tail = reshape(permutedims(targets, (3, 1, 2))[1:max(1, L) - 1, :, :], max(0, L - 1), B * K)
    decoder_input = vcat(fill(start_token, 1, B * K), input_tail)
    decoder_target = reshape(permutedims(targets, (3, 1, 2)), L, B * K)
    # Collate pads with 0; map to pad token so embedding gets a valid index and loss masks correctly
    decoder_input = ifelse.(decoder_input .== 0, pad_token, decoder_input)
    decoder_target = ifelse.(decoder_target .== 0, pad_token, decoder_target)
    # (B, K) with [b,k] = k <= n_stations[b]; vec is column-major so B varies fast,
    # matching the (L, B, K) -> (L, B*K) reshape used for decoder_input/decoder_target.
    station_mask = vec(reshape(n_stations, :, 1) .>= reshape(1:K, 1, :))
    # (1, B*K): station index per column, B varies fast to match decoder ordering.
    station_ids = reshape(repeat(reshape(1:K, 1, 1, K), 1, B, 1), 1, B * K)
    (; decoder_input, decoder_target, station_mask, station_ids)
end

"""
    repeat_memory_for_stations(memory, max_stations)

`memory` (dim, enc_len, batch) -> (dim, enc_len, batch × max_stations) by repeating
each batch item max_stations times. Pure (no mutation of memory).
"""
function repeat_memory_for_stations(memory::AbstractArray{T,3}, max_stations::Int) where T
    # (d, enc_len, B) -> (d, enc_len, B*K) by repeating along the batch dimension (Zygote-friendly)
    repeat(memory, 1, 1, max_stations)
end

# ─── Loss (masked cross-entropy over stations) ─────────────────────────────────

"""
    multi_station_cross_entropy(logits, decoder_target, station_mask; pad_idx)

Logits (vocab, seq, batch×K), decoder_target (seq, batch×K), station_mask (batch×K).
Mean cross-entropy over valid (b,k) positions only. PAD is masked out (batch alignment).
EOS is not masked — the model is trained to predict EOS after the last character so it
learns when to stop (alignment with encoder content/silence).
"""
function multi_station_cross_entropy(
    logits::AbstractArray{T,3},
    decoder_target::AbstractArray{<:Integer,2},
    station_mask::AbstractVector{Bool};
    pad_idx::Int = PAD_TOKEN_IDX,
) where T
    vocab, seq_len, batch_k = size(logits)
    log_probs = Flux.logsoftmax(logits; dims=1)
    nll_flat = -sum(Flux.onehotbatch(vec(decoder_target), 1:vocab) .* reshape(log_probs, vocab, :); dims=1)
    nll = reshape(nll_flat, seq_len, batch_k)
    valid = (decoder_target .!= pad_idx) .& reshape(station_mask, 1, batch_k)
    total_valid = max(sum(valid), 1)
    sum(nll .* valid) / total_valid
end

# ─── Training step (one batch) ───────────────────────────────────────────────

"""
    prepare_training_batch(batch) -> (spec, decoder_input, decoder_target, station_mask, station_ids)

Prepare tensors from a Batch. For CUDA, move each to GPU then call train_step.
"""
function prepare_training_batch(batch::Batch)
    dec = prepare_decoder_batch(batch.targets, batch.n_stations)
    (batch.spectrogram, dec.decoder_input, dec.decoder_target, dec.station_mask, dec.station_ids)
end

"""
    train_step(model, spec, decoder_input, decoder_target, station_mask, station_ids; encoder_dropout, rng)

Single training step. All array arguments must already be on the desired device (e.g. gpu(...)).
If encoder_dropout > 0, with that probability the encoder output is zeroed for the whole batch
so the decoder cannot rely on it; this encourages the decoder to actually use the encoder when present.
"""
function train_step(
    model::SpectrogramEncoderDecoder,
    spec,
    decoder_input,
    decoder_target,
    station_mask,
    station_ids;
    encoder_dropout::Real = 0.0,
    rng::AbstractRNG = Random.default_rng(),
)
    memory = model.encoder(spec)
    K = size(decoder_input, 2) ÷ size(memory, 3)
    memory_rep = repeat_memory_for_stations(memory, K)
    if encoder_dropout > 0 && rand(rng, Float32) < encoder_dropout
        memory_rep = memory_rep .* 0f0  # same device and shape, zeros
    end
    logits = model.decoder(decoder_input, memory_rep, station_ids)
    multi_station_cross_entropy(logits, decoder_target, station_mask)
end

train_step(model::SpectrogramEncoderDecoder, batch::Batch; kws...) =
    train_step(model, prepare_training_batch(batch)...; kws...)

# ─── Autoregressive sampling ───────────────────────────────────────────────────

"""
    decode_autoregressive(model, spec, n_stations; max_len, start_token, to_device)

Decode up to `n_stations` transcripts from one spectrogram. Stops early when every station
has emitted EOS (or at max_len). Returns (seq_len, n_stations) token indices.
to_device(x) should place x on the same device as spec (e.g. Flux.gpu or Flux.cpu); default identity.
"""
function decode_autoregressive(
    model::SpectrogramEncoderDecoder,
    spec,
    n_stations::Int;
    max_len::Int = 256,
    start_token::Int = START_TOKEN_IDX,
    to_device = identity,
)
    memory = model.encoder(spec)
    memory_rep = repeat_memory_for_stations(memory, n_stations)
    station_ids = to_device(reshape(collect(1:n_stations), 1, n_stations))
    ids_so_far = to_device(fill(start_token, 1, n_stations))
    done_per_station = falses(n_stations)

    for _ in 2:max_len
        logits = model.decoder(ids_so_far, memory_rep, station_ids)
        next_logits = selectdim(logits, 2, size(logits, 2))
        next_logits_cpu = Array(next_logits)
        next_logits_cpu[PAD_TOKEN_IDX, :] .= -1f10
        next_logits_cpu[START_TOKEN_IDX, :] .= -1f10
        am = argmax(next_logits_cpu; dims=1)
        next_ids_cpu = reshape(map(i -> i[1], am), 1, n_stations)
        for k in 1:n_stations
            next_ids_cpu[1, k] == EOS_TOKEN_IDX && (done_per_station[k] = true)
        end
        next_ids = to_device(next_ids_cpu)
        ids_so_far = cat(ids_so_far, next_ids; dims=1)
        all(done_per_station) && break
    end

    ids_so_far
end
