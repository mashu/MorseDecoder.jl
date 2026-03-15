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
using MorseSimulator: DatasetConfig, DirectPath, generate_sample as sim_generate_sample

# ─── Sample and config ───────────────────────────────────────────────────────

"""
A single training example from MorseSimulator.

- `spectrogram` : (n_mels × n_frames) Float32 — mel spectrogram in 200–900 Hz band.
- `token_ids`   : target sequence (label_to_token_ids(sample.label)).
"""
struct Sample
    spectrogram::Matrix{Float32}
    token_ids::Vector{Int}
end

"""
Configuration for training data.

- `dataset_config` : MorseSimulator.DatasetConfig (band 200–900 Hz, fft/hop for ~10 Hz and 50 WPM).
- `max_frames`     : unused by generate_sample; chunk size is passed to ChunkedBatchSource / ChunkedSampleSource.
"""
struct SamplerConfig
    dataset_config::DatasetConfig
    max_frames::Union{Int,Nothing}
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
    max_frames::Union{Int,Nothing} = 512,
)
    dataset_config = DatasetConfig(;
        path, sample_rate, fft_size, hop_size,
        n_mels, f_min, f_max, stations,
    )
    SamplerConfig(dataset_config, max_frames)
end

# ─── Single sample ───────────────────────────────────────────────────────────

"""
    generate_sample(cfg; rng) → Sample

One full conversation from MorseSimulator (spectrogram + label as token_ids).
No truncation: chunked training uses ChunkedSampleSource / ChunkedBatchSource with a
max_frames chunk size instead. For bounded-length batches use generate_chunked_batch or
iterate ChunkedBatchSource(cfg, batch_size, max_frames; rng).
"""
function generate_sample(cfg::SamplerConfig; rng::AbstractRNG = Random.default_rng())
    ds = sim_generate_sample(rng, cfg.dataset_config)
    spec = Float32.(ds.mel_spectrogram)
    ids = label_to_token_ids(ds.label)
    Sample(spec, ids)
end

# ─── Batch collation ────────────────────────────────────────────────────────

"""
Padded batch for the network.

- `spectrogram`    : (n_bins, batch, max_time)
- `targets`        : (batch, max_seq) — token IDs, zero-padded
- `target_lengths` : (batch,)
- `input_lengths`  : (batch,) — spectrogram time length per sample
- `n_stations`     : (batch,) — from metadata when available
- `frequencies`    : (batch, max_ns) — not used with simulator, zeros
"""
struct Batch
    spectrogram::Array{Float32,3}
    targets::Array{Int,2}
    target_lengths::Vector{Int}
    input_lengths::Vector{Int}
    n_stations::Vector{Int}
    frequencies::Matrix{Float32}
end

function collate(samples::AbstractVector{Sample})
    B = length(samples)
    max_time = maximum(size(s.spectrogram, 2) for s in samples)
    n_bins = size(first(samples).spectrogram, 1)
    max_seq = maximum(length(s.token_ids) for s in samples)

    spec_batch = zeros(Float32, n_bins, B, max_time)
    tgt_batch = zeros(Int, B, max_seq)
    tgt_lens = zeros(Int, B)
    in_lens = zeros(Int, B)
    ns_vec = zeros(Int, B)
    freq_batch = zeros(Float32, B, 6)

    @inbounds for (b, s) in enumerate(samples)
        T = size(s.spectrogram, 2)
        spec_batch[:, b, 1:T] .= s.spectrogram
        in_lens[b] = T
        tgt_lens[b] = length(s.token_ids)
        tgt_batch[b, 1:tgt_lens[b]] .= s.token_ids
    end

    Batch(spec_batch, tgt_batch, tgt_lens, in_lens, ns_vec, freq_batch)
end

"""
    generate_batch(cfg, batch_size; rng, parallel) → Batch
    generate_batch_fast(cfg, batch_size; rng, parallel) → Batch

Generate and collate a batch using MorseSimulator. Both names supported.
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

generate_batch_fast(cfg::SamplerConfig, batch_size::Int; rng::AbstractRNG = Random.default_rng(), parallel::Bool = Threads.nthreads() > 1) =
    generate_batch(cfg, batch_size; rng, parallel)

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

"""
    segments_to_chunks(spec, token_ids, max_frames) -> Vector{Sample}

Split one long (spec, token_ids) into training samples with ≤ max_frames, using
transmission boundaries ([TS]..[TE]). Proportional mapping: token range (a,b) maps to
frames round((a-1)*T/L)+1 : round(b*T/L). If a segment is longer than max_frames,
subdivide its frame range into max_frames-sized windows and assign token spans proportionally
(so we may cut mid-word within a long transmission, but never at [TS]/[TE]).
Each chunk's token sequence is wrapped as START + segment + EOS for a valid decoder sequence.
Uses dispatch for spec slice: no conversion when spec is already Matrix{Float32}.
"""
chunk_spec_slice(spec::Matrix{Float32}, r) = spec[:, r]
chunk_spec_slice(spec::AbstractMatrix, r) = Float32.(spec[:, r])

function segments_to_chunks(spec::AbstractMatrix, token_ids::Vector{Int}, max_frames::Int)
    T = size(spec, 2)
    L = length(token_ids)
    L == 0 && return Sample[]
    segs = transmission_segments(token_ids)
    if isempty(segs)
        segs = [(1, L)]
    end
    out = Sample[]
    for (tok_start, tok_end) in segs
        n_tok = tok_end - tok_start + 1
        f_start = max(1, round(Int, (tok_start - 1) * T / L) + 1)
        f_end = min(T, round(Int, tok_end * T / L))
        n_frames = f_end - f_start + 1
        seg_tokens = token_ids[tok_start:tok_end]
        chunk_tokens = [START_TOKEN_IDX; seg_tokens; EOS_TOKEN_IDX]
        if n_frames <= max_frames
            push!(out, Sample(chunk_spec_slice(spec, f_start:f_end), chunk_tokens))
        else
            n_chunks = cld(n_frames, max_frames)
            for c in 1:n_chunks
                cf_start = f_start + (c - 1) * max_frames
                cf_end = min(f_end, f_start + c * max_frames - 1)
                t1 = tok_start + round(Int, (c - 1) * n_tok / n_chunks)
                t2 = (c == n_chunks) ? tok_end : (tok_start + round(Int, c * n_tok / n_chunks) - 1)
                t1 = max(tok_start, t1)
                t2 = min(tok_end, t2)
                if t2 >= t1
                    sub_tokens = [START_TOKEN_IDX; token_ids[t1:t2]; EOS_TOKEN_IDX]
                    push!(out, Sample(chunk_spec_slice(spec, cf_start:cf_end), sub_tokens))
                end
            end
        end
    end
    out
end

# ─── Chunked iterators (conversation → chunks → batches) ───────────────────────
# Standard iteration interface: iterate(iter), iterate(iter, state), IteratorSize,
# IteratorEltype, eltype (https://docs.julialang.org/en/v1/manual/interfaces/#Iteration).

"""
    ChunkedConversation(spec, token_ids, max_frames)

Iterable over `Sample` chunks of one conversation. Splits at [TS]/[TE] boundaries
so chunks never cut mid-turn; each chunk has ≤ max_frames. Use when you have a
full (spec, token_ids) and want to process or decode it chunk by chunk (e.g. streaming).
"""
struct ChunkedConversation{S,T}
    spec::S
    token_ids::T
    max_frames::Int
end

function Base.iterate(cc::ChunkedConversation)
    chunks = segments_to_chunks(cc.spec, cc.token_ids, cc.max_frames)
    isempty(chunks) ? nothing : (chunks[1], (chunks, 2))
end

function Base.iterate(cc::ChunkedConversation, state)
    chunks, i = state
    i > length(chunks) ? nothing : (chunks[i], (chunks, i + 1))
end

Base.IteratorSize(::Type{<:ChunkedConversation}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:ChunkedConversation}) = Base.HasEltype()
Base.eltype(::Type{<:ChunkedConversation}) = Sample

"""
    ChunkedSampleSource(cfg, max_frames; rng)

Infinite iterator of `Sample` chunks. Generates full conversations via the simulator,
splits them with `ChunkedConversation`, and yields chunks; when a conversation is
exhausted, the next is generated. Use as the source for batched training.
"""
mutable struct ChunkedSampleSource{C,R}
    cfg::C
    max_frames::Int
    rng::R
end

ChunkedSampleSource(cfg, max_frames::Int; rng::AbstractRNG = Random.default_rng()) =
    ChunkedSampleSource(cfg, max_frames, rng)

function Base.iterate(src::ChunkedSampleSource)
    s = generate_sample(src.cfg; rng = src.rng)
    chunks = segments_to_chunks(s.spectrogram, s.token_ids, src.max_frames)
    while isempty(chunks)
        s = generate_sample(src.cfg; rng = src.rng)
        chunks = segments_to_chunks(s.spectrogram, s.token_ids, src.max_frames)
    end
    (chunks[1], (chunks, 2))
end

function Base.iterate(src::ChunkedSampleSource, state)
    chunks, i = state
    if i <= length(chunks)
        return (chunks[i], (chunks, i + 1))
    end
    s = generate_sample(src.cfg; rng = src.rng)
    chunks = segments_to_chunks(s.spectrogram, s.token_ids, src.max_frames)
    while isempty(chunks)
        s = generate_sample(src.cfg; rng = src.rng)
        chunks = segments_to_chunks(s.spectrogram, s.token_ids, src.max_frames)
    end
    (chunks[1], (chunks, 2))
end

Base.IteratorSize(::Type{<:ChunkedSampleSource}) = Base.IsInfinite()
Base.IteratorEltype(::Type{<:ChunkedSampleSource}) = Base.HasEltype()
Base.eltype(::Type{<:ChunkedSampleSource}) = Sample

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

# ─── Convenience constructors (same types as above) ───────────────────────────

"""
    chunked_samples(cfg [ , rng ]; max_frames=512)

Return a `ChunkedSampleSource` (infinite iterator of `Sample` chunks). Each chunk has
≤ max_frames and is split at [TS]/[TE] boundaries.
"""
chunked_samples(cfg::SamplerConfig, rng::AbstractRNG = Random.default_rng(); max_frames::Int = 512) =
    ChunkedSampleSource(cfg, max_frames; rng)

"""
    generate_chunked_batch(cfg, batch_size, rng; max_frames=512) -> Batch

One batch of chunked samples: takes batch_size from a ChunkedBatchSource and collates.
"""
function generate_chunked_batch(cfg::SamplerConfig, batch_size::Int, rng::AbstractRNG; max_frames::Int = 512)
    batch, _ = iterate(ChunkedBatchSource(cfg, batch_size, max_frames; rng))
    batch
end
