"""
    sampler.jl — Training samples via MorseSimulator.jl.

Uses MorseSimulator's DatasetConfig with 200–900 Hz mel band, ~10 Hz frequency
resolution (to separate signals), and sufficient time resolution for dots/dashes
up to 50 WPM. Produces (spectrogram, token_ids) pairs; token_ids use vocab.jl
(<START>, <END>, [S1]..[S6], [TS], [TE], chars).

Chunked training: use `chunked_samples` or `generate_chunked_batch` to iterate over
conversations split at [TS]/[TE] boundaries so no chunk cuts mid-turn. Each yielded
sample has ≤ max_frames and matching (spec_slice, transcript_slice).
"""

using Random
using MorseSimulator: DatasetConfig, DirectPath, generate_sample as sim_generate_sample,
    AbstractTokenTiming, TokenTiming, NoTiming

# ─── Sample (parametric on timing type for type stability) ───────────────────

"""
A single training example from MorseSimulator.

- `spectrogram` : (n_mels × n_frames) Float32 — mel spectrogram in 200–900 Hz band.
- `token_ids`   : target sequence (label_to_token_ids(sample.label)).
- `token_timing` : TokenTiming from simulator (exact chunk alignment) or NoTiming (no chunks yielded).
"""
struct Sample{TT<:AbstractTokenTiming}
    spectrogram::Matrix{Float32}
    token_ids::Vector{Int}
    token_timing::TT
end

Sample(spec::Matrix{Float32}, ids::Vector{Int}) = Sample(spec, ids, NoTiming())

# ─── SamplerConfig ───────────────────────────────────────────────────────────

"""
Configuration for training data.

`dataset_config` : MorseSimulator.DatasetConfig (band 200–900 Hz, fft/hop for ~10 Hz and 50 WPM).
Chunk size (max_frames) is passed directly to ChunkedBatchSource / ChunkedSampleSource.
"""
struct SamplerConfig
    dataset_config::DatasetConfig
end

# 200–900 Hz only (Morse CW). ~10 Hz freq resolution → fft_size ≥ sr/10 (44100/10=4410 → 4096).
# 50 WPM: dit ≈ 24 ms; ≥4–5 frames per dit → hop ≤ ~5 ms; hop 128 @ 44.1 kHz ≈ 2.9 ms (~8 frames/dot).
function SamplerConfig(;
    path = DirectPath(),
    sample_rate::Int = 44100,
    fft_size::Int = 4096,      # ~10.8 Hz bin at 44.1 kHz
    hop_size::Int = 128,       # ~2.9 ms/frame, 4–5+ frames/dot up to ~80 WPM
    n_mels::Int = 40,
    f_min::Float64 = 200.0,
    f_max::Float64 = 900.0,
    stations::UnitRange{Int} = 2:4,
)
    dataset_config = DatasetConfig(;
        path, sample_rate, fft_size, hop_size,
        n_mels, f_min, f_max, stations,
    )
    SamplerConfig(dataset_config)
end

# ─── Single sample ───────────────────────────────────────────────────────────

"""
    generate_sample(cfg; rng) → Sample

One full conversation from MorseSimulator (spectrogram + label as token_ids).
No truncation: chunked training uses ChunkedSampleSource / ChunkedBatchSource with a
max_frames chunk size instead.
"""
function generate_sample(cfg::SamplerConfig; rng::AbstractRNG = Random.default_rng())
    ds = sim_generate_sample(rng, cfg.dataset_config)
    spec = Float32.(ds.mel_spectrogram)
    ids = label_to_token_ids(ds.label)
    Sample(spec, ids, ds.token_timing)
end

# ─── Batch collation ────────────────────────────────────────────────────────

"""
Padded batch for the network.

- `spectrogram`    : (n_bins, batch, max_time)
- `targets`        : (batch, max_seq) — token IDs, zero-padded
- `target_lengths` : (batch,)
- `input_lengths`  : (batch,) — spectrogram time length per sample
"""
struct Batch
    spectrogram::Array{Float32,3}
    targets::Array{Int,2}
    target_lengths::Vector{Int}
    input_lengths::Vector{Int}
end

function collate(samples::AbstractVector{<:Sample})
    B = length(samples)
    max_time = maximum(size(s.spectrogram, 2) for s in samples)
    n_bins = size(first(samples).spectrogram, 1)
    max_seq = maximum(length(s.token_ids) for s in samples)

    spec_batch = zeros(Float32, n_bins, B, max_time)
    tgt_batch = zeros(Int, B, max_seq)
    tgt_lens = zeros(Int, B)
    in_lens = zeros(Int, B)

    @inbounds for (b, s) in enumerate(samples)
        T = size(s.spectrogram, 2)
        spec_batch[:, b, 1:T] .= s.spectrogram
        in_lens[b] = T
        tgt_lens[b] = length(s.token_ids)
        tgt_batch[b, 1:tgt_lens[b]] .= s.token_ids
    end

    Batch(spec_batch, tgt_batch, tgt_lens, in_lens)
end

# ─── Batch generation ────────────────────────────────────────────────────────

"""
    generate_batch(cfg, batch_size; rng, parallel) → Batch

Generate and collate a batch of full conversations using MorseSimulator.
"""
function generate_batch(cfg::SamplerConfig, batch_size::Int;
                       rng::AbstractRNG = Random.default_rng(),
                       parallel::Bool = Threads.nthreads() > 1)
    if parallel && Threads.nthreads() > 1
        base_seed = rand(rng, UInt)
        samples = Vector{Sample}(undef, batch_size)
        Threads.@threads for i in 1:batch_size
            rng_i = MersenneTwister(hash(base_seed, UInt(i)))
            @inbounds samples[i] = generate_sample(cfg; rng = rng_i)
        end
        collate(samples)
    else
        collate([generate_sample(cfg; rng) for _ in 1:batch_size])
    end
end

# ─── Chunked training: iterable conversation → chunks (≤ max_frames) ───────────

"""
    transmission_segments(token_ids) -> Vector{Tuple{Int,Int}}

Parse token_ids into (tok_start, tok_end) ranges for each transmission: from [TS] to [TE] (inclusive).
Returns empty vector if no [TS]/[TE] pairs found.
"""
function transmission_segments(token_ids::Vector{Int})
    out = Tuple{Int,Int}[]
    i = 1
    while i <= length(token_ids)
        if token_ids[i] == TS_TOKEN_IDX
            tok_start = i
            j = i
            while j <= length(token_ids) && token_ids[j] != TE_TOKEN_IDX
                j += 1
            end
            if j <= length(token_ids)
                push!(out, (tok_start, j))
                i = j + 1
            else
                break
            end
        else
            i += 1
        end
    end
    out
end

# ─── Chunking: exact alignment from simulator timings only ─────────────────────
# No proportional fallback; simulator provides TokenTiming. NoTiming => no chunks.

"""Frame range for a transmission segment — exact alignment from simulator timings."""
function segment_frame_range(tok_start::Int, tok_end::Int, n_total_frames::Int, ::Int, timing::TokenTiming)
    f_start = clamp(timing.token_start_frames[tok_start], 1, n_total_frames)
    f_end = clamp(timing.token_end_frames[tok_end], 1, n_total_frames)
    (f_start, f_end)
end

"""Token range for a sub-chunk — exact overlap rule from simulator timings."""
function subchunk_token_range(tok_start::Int, tok_end::Int, ::Int, ::Int,
                              cf_start::Int, cf_end::Int, timing::TokenTiming)
    st = timing.token_start_frames
    en = timing.token_end_frames
    t1 = tok_start
    while t1 <= tok_end && en[t1] < cf_start
        t1 += 1
    end
    t2 = tok_end
    while t2 >= tok_start && st[t2] > cf_end
        t2 -= 1
    end
    (t1, t2)
end

# ─── Spec slice dispatch ────────────────────────────────────────────────────

chunk_spec_slice(spec::Matrix{Float32}, r) = spec[:, r]
chunk_spec_slice(spec::AbstractMatrix, r) = Float32.(spec[:, r])

# ─── Main chunking function (single method, dispatches on timing) ───────────

"""
    segments_to_chunks(spec, token_ids, max_frames, timing) -> Vector{Sample}

Split one long (spec, token_ids) into training samples with ≤ max_frames, using
transmission boundaries ([TS]..[TE]). Uses **exact** token-frame alignment from
simulator timings (TokenTiming) only. When timing is NoTiming() (e.g. simulator
alignment failed), returns [] so that sample is skipped — no proportional fallback.
Each chunk's token sequence is wrapped as START + segment + EOS.
"""
segments_to_chunks(spec::AbstractMatrix, token_ids::Vector{Int}, max_frames::Int, ::NoTiming) =
    Sample{NoTiming}[]

function segments_to_chunks(spec::AbstractMatrix, token_ids::Vector{Int},
                            max_frames::Int, timing::TokenTiming)
    T_frames = size(spec, 2)
    L = length(token_ids)
    L == 0 && return Sample{NoTiming}[]
    segs = transmission_segments(token_ids)
    isempty(segs) && (segs = [(1, L)])

    out = Sample{NoTiming}[]
    for (tok_start, tok_end) in segs
        f_start, f_end = segment_frame_range(tok_start, tok_end, T_frames, L, timing)
        n_frames = f_end - f_start + 1
        n_frames <= 0 && continue  # skip invalid alignment (e.g. timing gives f_start > f_end)
        seg_tokens = token_ids[tok_start:tok_end]
        chunk_tokens = [START_TOKEN_IDX; seg_tokens; EOS_TOKEN_IDX]

        if n_frames <= max_frames
            push!(out, Sample(chunk_spec_slice(spec, f_start:f_end), chunk_tokens))
        else
            n_chunks = cld(n_frames, max_frames)
            for c in 1:n_chunks
                cf_start = f_start + (c - 1) * max_frames
                cf_end = min(f_end, f_start + c * max_frames - 1)
                t1, t2 = subchunk_token_range(tok_start, tok_end, c, n_chunks, cf_start, cf_end, timing)
                if t2 >= t1
                    sub_tokens = [START_TOKEN_IDX; token_ids[t1:t2]; EOS_TOKEN_IDX]
                    push!(out, Sample(chunk_spec_slice(spec, cf_start:cf_end), sub_tokens))
                end
            end
        end
    end
    out
end

# No timing => no chunks (caller can use this when timing is not available)
segments_to_chunks(spec::AbstractMatrix, token_ids::Vector{Int}, max_frames::Int) =
    segments_to_chunks(spec, token_ids, max_frames, NoTiming())

# ─── Chunked iterators (conversation → chunks → batches) ───────────────────────
# Standard iteration interface: iterate(iter), iterate(iter, state), IteratorSize,
# IteratorEltype, eltype (https://docs.julialang.org/en/v1/manual/interfaces/#Iteration).

"""
    ChunkedConversation(spec, token_ids, max_frames; timing=NoTiming())
    ChunkedConversation(sample::Sample, max_frames)

Iterable over `Sample` chunks of one conversation. Splits at [TS]/[TE] boundaries
so chunks never cut mid-turn; each chunk has ≤ max_frames.
Uses exact token-frame alignment from simulator (TokenTiming) only; with NoTiming
yields no chunks. Prefer `ChunkedConversation(sample, max_frames)` when you have a Sample.
"""
struct ChunkedConversation{S,T}
    spec::S
    token_ids::T
    max_frames::Int
    timing::AbstractTokenTiming
end

ChunkedConversation(spec, token_ids, max_frames::Int; timing::AbstractTokenTiming=NoTiming()) =
    ChunkedConversation(spec, token_ids, max_frames, timing)
ChunkedConversation(sample::Sample, max_frames::Int) =
    ChunkedConversation(sample.spectrogram, sample.token_ids, max_frames, sample.token_timing)

function Base.iterate(cc::ChunkedConversation)
    chunks = segments_to_chunks(cc.spec, cc.token_ids, cc.max_frames, cc.timing)
    isempty(chunks) ? nothing : (chunks[1], (chunks, 2))
end

function Base.iterate(::ChunkedConversation, state)
    chunks, i = state
    i > length(chunks) ? nothing : (chunks[i], (chunks, i + 1))
end

Base.IteratorSize(::Type{<:ChunkedConversation}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:ChunkedConversation}) = Base.HasEltype()
Base.eltype(::Type{<:ChunkedConversation}) = Sample{NoTiming}

"""
    ChunkedSampleSource(cfg, max_frames; rng)

Infinite iterator of `Sample` chunks. Generates full conversations via the simulator,
splits them with timing-aware chunking, and yields chunks one at a time.
"""
mutable struct ChunkedSampleSource{C,R}
    cfg::C
    max_frames::Int
    rng::R
end

ChunkedSampleSource(cfg, max_frames::Int; rng::AbstractRNG = Random.default_rng()) =
    ChunkedSampleSource(cfg, max_frames, rng)

# Shared logic: generate conversations until we get non-empty chunks, then yield next chunk.
function next_chunk(src::ChunkedSampleSource, chunks, i::Int)
    if i <= length(chunks)
        return (chunks[i], (chunks, i + 1))
    end
    # Current conversation exhausted; generate new one
    while true
        s = generate_sample(src.cfg; rng = src.rng)
        new_chunks = segments_to_chunks(s.spectrogram, s.token_ids, src.max_frames, s.token_timing)
        if !isempty(new_chunks)
            return (new_chunks[1], (new_chunks, 2))
        end
    end
end

function Base.iterate(src::ChunkedSampleSource)
    next_chunk(src, Sample{NoTiming}[], 1)  # empty chunks forces generation
end

function Base.iterate(src::ChunkedSampleSource, state)
    chunks, i = state
    next_chunk(src, chunks, i)
end

Base.IteratorSize(::Type{<:ChunkedSampleSource}) = Base.IsInfinite()
Base.IteratorEltype(::Type{<:ChunkedSampleSource}) = Base.HasEltype()
Base.eltype(::Type{<:ChunkedSampleSource}) = Sample{NoTiming}

"""
    ChunkedBatchSource(cfg, batch_size, max_frames; rng)

Infinite iterator of `Batch`. Each batch is built by taking `batch_size` chunks from
a `ChunkedSampleSource` and collating. Use for training: `for batch in ChunkedBatchSource(...)`.
"""
struct ChunkedBatchSource{C,R}
    cfg::C
    batch_size::Int
    max_frames::Int
    rng::R
end

ChunkedBatchSource(cfg, batch_size::Int, max_frames::Int; rng::AbstractRNG = Random.default_rng()) =
    ChunkedBatchSource(cfg, batch_size, max_frames, rng)

function Base.iterate(src::ChunkedBatchSource)
    stream = ChunkedSampleSource(src.cfg, src.max_frames; rng = src.rng)
    samples = collect(Iterators.take(stream, src.batch_size))
    (collate(samples), stream)
end

function Base.iterate(src::ChunkedBatchSource, stream::ChunkedSampleSource)
    samples = collect(Iterators.take(stream, src.batch_size))
    (collate(samples), stream)
end

Base.IteratorSize(::Type{<:ChunkedBatchSource}) = Base.IsInfinite()
Base.IteratorEltype(::Type{<:ChunkedBatchSource}) = Base.HasEltype()
Base.eltype(::Type{<:ChunkedBatchSource}) = Batch

# ─── Convenience constructors ─────────────────────────────────────────────────

"""
    chunked_samples(cfg [, rng]; max_frames=512)

Return a `ChunkedSampleSource` (infinite iterator of `Sample` chunks).
"""
chunked_samples(cfg::SamplerConfig, rng::AbstractRNG = Random.default_rng(); max_frames::Int = 512) =
    ChunkedSampleSource(cfg, max_frames; rng)

"""
    generate_chunked_batch(cfg, batch_size, rng; max_frames=512) -> Batch

One batch of chunked samples.
"""
function generate_chunked_batch(cfg::SamplerConfig, batch_size::Int, rng::AbstractRNG; max_frames::Int = 512)
    batch, _ = iterate(ChunkedBatchSource(cfg, batch_size, max_frames; rng))
    batch
end
