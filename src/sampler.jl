"""
    sampler.jl — Training samples via MorseSimulator.jl with continuation-aware chunking.

Conversations are 10k–20k+ frames. Chunking splits at [TS]/[TE] boundaries into
≤ max_frames pieces. Critically, chunks are **continuations** of the same conversation:

  Chunk 1: tokens = [START, [TS], [S1], text, [TE]]        prefix = []
  Chunk 2: tokens = [[TS], [S2], text, [TE]]               prefix = chunk1 tokens
  Chunk N: tokens = [[TS], [SN], text, [TE], EOS]           prefix = last max_prefix tokens

The decoder sees [prefix ++ tokens] during teacher forcing, but loss is computed only
on the current chunk's tokens (prefix positions are masked). This teaches the model
that chunks are continuations, matching decode_conversation's inference behavior.

Data layout:
  Batch.spectrogram  : (n_bins, batch, max_time) — this chunk's spectrogram only
  Batch.targets      : (batch, max_seq) — prefix ++ chunk_tokens, zero-padded
  Batch.prefix_lengths : how many prefix tokens per sample (loss-masked)

Attention operates over time within each spectrogram independently. Batch dimension
is for GPU parallelism only — spectrograms never mix across batch elements.
"""

using Random
using MorseSimulator: DatasetConfig, DirectPath, generate_sample as sim_generate_sample,
    AbstractTokenTiming, TokenTiming, NoTiming

# ─── Sample (for inference) ──────────────────────────────────────────────────

"""
A single example from MorseSimulator: mel spectrogram + token IDs + optional timing.
Used for inference (decode_conversation). Training uses TrainingChunk instead.
"""
struct Sample{TT<:AbstractTokenTiming}
    spectrogram::Matrix{Float32}
    token_ids::Vector{Int}
    token_timing::TT
end

Sample(spec::Matrix{Float32}, ids::Vector{Int}) = Sample{NoTiming}(spec, ids, NoTiming())
Sample(spec::Matrix{Float32}, ids::Vector{Int}, timing::NoTiming) = Sample{NoTiming}(spec, ids, timing)
Sample(spec::Matrix{Float32}, ids::Vector{Int}, timing::TokenTiming) = Sample{TokenTiming}(spec, ids, timing)

# ─── TrainingChunk (for continuation-aware training) ─────────────────────────

"""
One chunk of a conversation, carrying context from previous chunks.

- `spectrogram` : (n_mels, n_frames) — this chunk's audio
- `token_ids`   : this chunk's target tokens (START only on first, EOS only on last)
- `prefix`      : teacher-forced context from previous chunks (capped at max_prefix)
"""
struct TrainingChunk
    spectrogram::Matrix{Float32}
    token_ids::Vector{Int}
    prefix::Vector{Int}
end

# ─── Single sample from simulator ────────────────────────────────────────────

"""
    generate_sample(cfg::DatasetConfig; rng) → Sample

One full conversation from MorseSimulator (spectrogram + token_ids).
"""
function generate_sample(cfg::DatasetConfig; rng::AbstractRNG = Random.default_rng())
    ds = sim_generate_sample(rng, cfg)
    Sample(Float32.(ds.spectrogram), label_to_token_ids(ds.label), ds.token_timing)
end

# ─── Batch ───────────────────────────────────────────────────────────────────

"""
Padded batch for the network.

- `spectrogram`     : (n_bins, batch, max_time) — zero-padded
- `targets`         : (batch, max_seq) — [prefix ++ chunk_tokens], zero-padded
- `target_lengths`  : total length (prefix + chunk tokens) per sample
- `prefix_lengths`  : prefix length per sample (loss-masked during training)
- `input_lengths`   : spectrogram time frames per sample
"""
struct Batch
    spectrogram::Array{Float32,3}
    targets::Array{Int,2}
    target_lengths::Vector{Int}
    prefix_lengths::Vector{Int}
    input_lengths::Vector{Int}
end

function collate(chunks::AbstractVector{TrainingChunk})
    B = length(chunks)
    n_bins = size(first(chunks).spectrogram, 1)
    max_time = maximum(size(c.spectrogram, 2) for c in chunks)
    max_seq = maximum(length(c.prefix) + length(c.token_ids) for c in chunks)

    spec = zeros(Float32, n_bins, B, max_time)
    tgt = zeros(Int, B, max_seq)
    tgt_lens = Vector{Int}(undef, B)
    pfx_lens = Vector{Int}(undef, B)
    in_lens = Vector{Int}(undef, B)

    @inbounds for (b, c) in enumerate(chunks)
        T = size(c.spectrogram, 2)
        spec[:, b, 1:T] .= c.spectrogram
        in_lens[b] = T

        P = length(c.prefix)
        L = length(c.token_ids)
        tgt[b, 1:P] .= c.prefix
        tgt[b, P+1:P+L] .= c.token_ids
        tgt_lens[b] = P + L
        pfx_lens[b] = P
    end

    Batch(spec, tgt, tgt_lens, pfx_lens, in_lens)
end

# ─── Chunking: continuation-aware split at [TS]/[TE] boundaries ─────────────
#
# [TS] = transmission start (station starts keying), [TE] = transmission end (station unkeys).
# There are many [TS]/[TE] because each station turn is one [TS]…[TE] segment (S1 talks, then S2, then S1, …).
# Chunk boundaries: normally between [TE] and the next [TS] (gap/silence). Only when a single
# transmission is longer than max_frames do we sub-chunk it — then a boundary can cut through the signal.

"""
    transmission_segments(token_ids) -> Vector{Tuple{Int,Int}}

Find (start, end) index pairs for each [TS]…[TE] transmission in the token sequence.
"""
function transmission_segments(token_ids::Vector{Int})
    segs = Tuple{Int,Int}[]
    i = 1
    while i <= length(token_ids)
        if token_ids[i] == TS_TOKEN_IDX
            j = i
            while j <= length(token_ids) && token_ids[j] != TE_TOKEN_IDX
                j += 1
            end
            j <= length(token_ids) && push!(segs, (i, j))
            i = j + 1
        else
            i += 1
        end
    end
    segs
end

"""Take last n elements of a vector (or all if shorter)."""
tail(v::Vector, n::Int) = length(v) <= n ? copy(v) : v[end-n+1:end]

"""
    chunk_conversation(sample, max_frames; max_prefix=256) -> Vector{TrainingChunk}

Split a conversation into continuation-aware training chunks ≤ max_frames.
Only the first chunk gets START, only the last gets EOS. Each chunk carries
a prefix of accumulated tokens from all previous chunks (capped at max_prefix).

Dispatches on timing: TokenTiming gives exact frame alignment, NoTiming returns [].
"""
chunk_conversation(::Sample{NoTiming}, ::Int; max_prefix::Int = 256) = TrainingChunk[]

function chunk_conversation(sample::Sample{TokenTiming}, max_frames::Int;
                            max_prefix::Int = 256)
    spec, ids, timing = sample.spectrogram, sample.token_ids, sample.token_timing
    T_total = size(spec, 2)
    isempty(ids) && return TrainingChunk[]

    segs = transmission_segments(ids)
    isempty(segs) && return TrainingChunk[]

    chunks = TrainingChunk[]
    accumulated = Int[]   # all tokens emitted by previous chunks

    for (seg_idx, (tok_start, tok_end)) in enumerate(segs)
        f_start = clamp(timing.token_start_frames[tok_start], 1, T_total)
        f_end = clamp(timing.token_end_frames[tok_end], 1, T_total)
        n_frames = f_end - f_start + 1
        n_frames <= 0 && continue

        seg_tokens = ids[tok_start:tok_end]
        is_first = isempty(accumulated)
        is_last = seg_idx == length(segs)

        if n_frames <= max_frames
            # One chunk per transmission; boundary falls in the gap after [TE] (before next [TS]) — no cut through signal.
            toks = Int[]
            is_first && push!(toks, START_TOKEN_IDX)
            append!(toks, seg_tokens)
            is_last && push!(toks, EOS_TOKEN_IDX)

            push!(chunks, TrainingChunk(Float32.(spec[:, f_start:f_end]),
                                        toks, tail(accumulated, max_prefix)))
            append!(accumulated, toks)
        else
            # Sub-chunk one long transmission (> max_frames): boundaries cut through the signal (middle of one station keying).
            # Assign tokens to sub-chunks by frame midpoint.
            n_sub = cld(n_frames, max_frames)
            for c in 1:n_sub
                cf_start = f_start + (c - 1) * max_frames
                cf_end = min(f_end, f_start + c * max_frames - 1)

                sub_ids = [ids[t] for t in tok_start:tok_end
                           if cf_start <= div(timing.token_start_frames[t] +
                                              timing.token_end_frames[t], 2) <= cf_end]
                isempty(sub_ids) && continue

                toks = Int[]
                is_first && c == 1 && push!(toks, START_TOKEN_IDX)
                append!(toks, sub_ids)
                is_last && c == n_sub && push!(toks, EOS_TOKEN_IDX)

                push!(chunks, TrainingChunk(Float32.(spec[:, cf_start:cf_end]),
                                            toks, tail(accumulated, max_prefix)))
                append!(accumulated, toks)
            end
        end
    end
    chunks
end

# ─── Batch generation ────────────────────────────────────────────────────────

"""
    generate_training_batch(cfg, batch_size, max_frames; rng, max_prefix=256) -> Batch

Generate one training batch of continuation-aware chunks.
"""
function generate_training_batch(cfg::DatasetConfig, batch_size::Int, max_frames::Int;
                                 rng::AbstractRNG = Random.default_rng(),
                                 max_prefix::Int = 256)
    buf = TrainingChunk[]
    while length(buf) < batch_size
        append!(buf, chunk_conversation(generate_sample(cfg; rng), max_frames;
                                        max_prefix))
    end
    collate(buf[1:batch_size])
end

# ─── Infinite batch iterator ─────────────────────────────────────────────────

"""
    BatchIterator(cfg, batch_size, max_frames; rng, max_prefix=256)

Infinite iterator yielding `Batch`. Refills an internal chunk buffer from the
simulator as needed.
"""
struct BatchIterator{R<:AbstractRNG}
    cfg::DatasetConfig
    batch_size::Int
    max_frames::Int
    max_prefix::Int
    rng::R
    buffer::Vector{TrainingChunk}
end

function BatchIterator(cfg::DatasetConfig, batch_size::Int, max_frames::Int;
                       rng::AbstractRNG = Random.default_rng(), max_prefix::Int = 256)
    BatchIterator(cfg, batch_size, max_frames, max_prefix, rng, TrainingChunk[])
end

function refill!(iter::BatchIterator)
    while length(iter.buffer) < iter.batch_size
        append!(iter.buffer, chunk_conversation(generate_sample(iter.cfg; rng = iter.rng),
                                                iter.max_frames; max_prefix = iter.max_prefix))
    end
end

function Base.iterate(iter::BatchIterator, ::Nothing = nothing)
    refill!(iter)
    (collate(splice!(iter.buffer, 1:iter.batch_size)), nothing)
end

Base.IteratorSize(::Type{<:BatchIterator}) = Base.IsInfinite()
Base.IteratorEltype(::Type{<:BatchIterator}) = Base.HasEltype()
Base.eltype(::Type{<:BatchIterator}) = Batch

# ─── ChunkedConversation (for decode_conversation) ──────────────────────────

"""
    ChunkedConversation(sample, max_frames)

Iterable of spectrogram matrices for `decode_conversation`. Splits at transmission
boundaries using timing. For NoTiming, splits uniformly (decode still works).
"""
struct ChunkedConversation
    spectrograms::Vector{Matrix{Float32}}
end

function ChunkedConversation(sample::Sample{TokenTiming}, max_frames::Int)
    tc = chunk_conversation(sample, max_frames)
    ChunkedConversation([c.spectrogram for c in tc])
end

function ChunkedConversation(sample::Sample{NoTiming}, max_frames::Int)
    spec = sample.spectrogram
    T = size(spec, 2)
    ChunkedConversation([Float32.(spec[:, (c-1)*max_frames+1:min(c*max_frames, T)])
                         for c in 1:cld(T, max_frames)])
end

function ChunkedConversation(spec, token_ids, max_frames::Int;
                             timing::AbstractTokenTiming = NoTiming())
    ChunkedConversation(Sample(Float32.(spec), token_ids, timing), max_frames)
end

Base.iterate(cc::ChunkedConversation) = iterate(cc.spectrograms)
Base.iterate(cc::ChunkedConversation, state) = iterate(cc.spectrograms, state)
Base.length(cc::ChunkedConversation) = length(cc.spectrograms)
Base.eltype(::Type{ChunkedConversation}) = Matrix{Float32}
