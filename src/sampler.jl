"""
    sampler.jl — Training samples via MorseSimulator.jl.

Produces (spectrogram, token_ids) pairs from MorseSimulator's DatasetConfig.
Chunked training splits conversations at [TS]/[TE] boundaries so no chunk
cuts mid-turn; each chunk has ≤ max_frames and matching transcript slice.

Data layout:
  Sample.spectrogram : (n_mels, n_frames) — one conversation or chunk
  Batch.spectrogram  : (n_bins, batch, max_time) — padded, ready for encoder
  Encoder expects    : (n_bins, batch, time) → internally (dim, time, batch)

Batch dimension is for GPU parallelism only — attention operates over the time
axis within each spectrogram independently. Spectrograms never mix across batch.
"""

using Random
using MorseSimulator: DatasetConfig, DirectPath, generate_sample as sim_generate_sample,
    AbstractTokenTiming, TokenTiming, NoTiming

# ─── Sample ──────────────────────────────────────────────────────────────────

"""
A single training example: mel spectrogram + token IDs + optional timing.

- `spectrogram` : (n_mels, n_frames) Float32
- `token_ids`   : target sequence
- `token_timing`: TokenTiming (exact alignment) or NoTiming (chunks unavailable)
"""
struct Sample{TT<:AbstractTokenTiming}
    spectrogram::Matrix{Float32}
    token_ids::Vector{Int}
    token_timing::TT
end

Sample(spec::Matrix{Float32}, ids::Vector{Int}) = Sample(spec, ids, NoTiming())

# ─── Single sample from simulator ────────────────────────────────────────────

"""
    generate_sample(cfg::DatasetConfig; rng) → Sample

One full conversation from MorseSimulator (spectrogram + token_ids).
"""
function generate_sample(cfg::DatasetConfig; rng::AbstractRNG = Random.default_rng())
    ds = sim_generate_sample(rng, cfg)
    Sample(Float32.(ds.mel_spectrogram), label_to_token_ids(ds.label), ds.token_timing)
end

# ─── Batch ───────────────────────────────────────────────────────────────────

"""
Padded batch for the network.

- `spectrogram`    : (n_bins, batch, max_time) — zero-padded
- `targets`        : (batch, max_seq) — token IDs, zero-padded
- `target_lengths` : per-sample target length
- `input_lengths`  : per-sample spectrogram time length
"""
struct Batch
    spectrogram::Array{Float32,3}
    targets::Array{Int,2}
    target_lengths::Vector{Int}
    input_lengths::Vector{Int}
end

function collate(samples::AbstractVector{<:Sample})
    B = length(samples)
    n_bins = size(first(samples).spectrogram, 1)
    max_time = maximum(size(s.spectrogram, 2) for s in samples)
    max_seq = maximum(length(s.token_ids) for s in samples)

    spec = zeros(Float32, n_bins, B, max_time)
    tgt = zeros(Int, B, max_seq)
    tgt_lens = Vector{Int}(undef, B)
    in_lens = Vector{Int}(undef, B)

    @inbounds for (b, s) in enumerate(samples)
        T = size(s.spectrogram, 2)
        spec[:, b, 1:T] .= s.spectrogram
        in_lens[b] = T
        L = length(s.token_ids)
        tgt_lens[b] = L
        tgt[b, 1:L] .= s.token_ids
    end

    Batch(spec, tgt, tgt_lens, in_lens)
end

# ─── Chunking: split conversation at [TS]/[TE] boundaries ───────────────────

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

"""
    chunk_conversation(sample, max_frames) -> Vector{Sample{NoTiming}}

Split a full conversation into training chunks ≤ max_frames at [TS]/[TE] boundaries.
Dispatches on timing type: TokenTiming gives exact frame alignment, NoTiming skips.
Each chunk's tokens are wrapped as [START, segment…, EOS].
"""
chunk_conversation(::Sample{NoTiming}, ::Int) = Sample{NoTiming}[]

function chunk_conversation(sample::Sample{TokenTiming}, max_frames::Int)
    spec, ids, timing = sample.spectrogram, sample.token_ids, sample.token_timing
    T_total = size(spec, 2)
    isempty(ids) && return Sample{NoTiming}[]

    segs = transmission_segments(ids)
    isempty(segs) && (segs = [(1, length(ids))])

    chunks = Sample{NoTiming}[]
    for (tok_start, tok_end) in segs
        f_start = clamp(timing.token_start_frames[tok_start], 1, T_total)
        f_end = clamp(timing.token_end_frames[tok_end], 1, T_total)
        n_frames = f_end - f_start + 1
        n_frames <= 0 && continue

        seg_tokens = [START_TOKEN_IDX; ids[tok_start:tok_end]; EOS_TOKEN_IDX]

        if n_frames <= max_frames
            push!(chunks, Sample(Float32.(spec[:, f_start:f_end]), seg_tokens))
        else
            # Split long transmission into sub-chunks; assign tokens by midpoint
            for c in 1:cld(n_frames, max_frames)
                cf_start = f_start + (c - 1) * max_frames
                cf_end = min(f_end, f_start + c * max_frames - 1)
                sub_ids = [ids[t] for t in tok_start:tok_end
                           if cf_start <= div(timing.token_start_frames[t] +
                                              timing.token_end_frames[t], 2) <= cf_end]
                isempty(sub_ids) && continue
                push!(chunks, Sample(Float32.(spec[:, cf_start:cf_end]),
                                     [START_TOKEN_IDX; sub_ids; EOS_TOKEN_IDX]))
            end
        end
    end
    chunks
end

# ─── Batch generation ────────────────────────────────────────────────────────

"""
    generate_training_batch(cfg, batch_size, max_frames; rng) -> Batch

Generate one training batch: create conversations until we have enough chunks.
"""
function generate_training_batch(cfg::DatasetConfig, batch_size::Int, max_frames::Int;
                                 rng::AbstractRNG = Random.default_rng())
    buf = Sample{NoTiming}[]
    while length(buf) < batch_size
        append!(buf, chunk_conversation(generate_sample(cfg; rng), max_frames))
    end
    collate(buf[1:batch_size])
end

# ─── Infinite batch iterator (for training loop with prefetch) ───────────────

"""
    BatchIterator(cfg, batch_size, max_frames; rng)

Infinite iterator yielding `Batch`. Refills an internal chunk buffer from the
simulator as needed. State is `nothing` — the buffer lives on the struct.
"""
struct BatchIterator{R<:AbstractRNG}
    cfg::DatasetConfig
    batch_size::Int
    max_frames::Int
    rng::R
    buffer::Vector{Sample{NoTiming}}
end

function BatchIterator(cfg::DatasetConfig, batch_size::Int, max_frames::Int;
                       rng::AbstractRNG = Random.default_rng())
    BatchIterator(cfg, batch_size, max_frames, rng, Sample{NoTiming}[])
end

function refill!(iter::BatchIterator)
    while length(iter.buffer) < iter.batch_size
        append!(iter.buffer, chunk_conversation(generate_sample(iter.cfg; rng=iter.rng),
                                                iter.max_frames))
    end
end

function Base.iterate(iter::BatchIterator, ::Nothing = nothing)
    refill!(iter)
    (collate(splice!(iter.buffer, 1:iter.batch_size)), nothing)
end

Base.IteratorSize(::Type{<:BatchIterator}) = Base.IsInfinite()
Base.IteratorEltype(::Type{<:BatchIterator}) = Base.HasEltype()
Base.eltype(::Type{<:BatchIterator}) = Batch

# ─── ChunkedConversation (decode_conversation compatibility) ─────────────────

"""
    ChunkedConversation(sample, max_frames)
    ChunkedConversation(spec, token_ids, max_frames; timing)

Iterable over chunks of one conversation for `decode_conversation`.
Pre-computes all chunks; yields `Sample` objects.
"""
struct ChunkedConversation
    chunks::Vector{Sample{NoTiming}}
end

function ChunkedConversation(sample::Sample, max_frames::Int)
    ChunkedConversation(chunk_conversation(sample, max_frames))
end

function ChunkedConversation(spec, token_ids, max_frames::Int;
                             timing::AbstractTokenTiming = NoTiming())
    ChunkedConversation(Sample(Float32.(spec), token_ids, timing), max_frames)
end

Base.iterate(cc::ChunkedConversation) =
    isempty(cc.chunks) ? nothing : (cc.chunks[1], 2)
Base.iterate(cc::ChunkedConversation, i::Int) =
    i > length(cc.chunks) ? nothing : (cc.chunks[i], i + 1)
Base.length(cc::ChunkedConversation) = length(cc.chunks)
Base.eltype(::Type{ChunkedConversation}) = Sample{NoTiming}
