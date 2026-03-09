"""
    sampler.jl — Training sample generation and real-time streaming.

Two modes:
  1. `generate_sample` / `generate_batch` — produce (spectrogram, labels) pairs
     for offline training.
  2. `BandStream` — stateful stream that yields chunks of spectrogram +
     per-station text labels, simulating a live band.

The network input is always a spectrogram matrix `(freq_bins × time_frames)`.
The targets are N text sequences (one per active station), to be decoded by
an encoder–decoder transformer with N output heads.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  Single training sample
# ═══════════════════════════════════════════════════════════════════════════════

"""
A single training example: spectrogram + per-station labels.

Fields:
- `spectrogram`  : (freq_bins × time_frames)  — network input
- `texts`        : one String per station      — target labels
- `frequencies`  : carrier Hz per station      — auxiliary info for the model
- `n_stations`   : number of active stations
"""
struct Sample
    spectrogram::Matrix{Float32}
    texts::Vector{String}
    frequencies::Vector{Float32}
    n_stations::Int
end

"""
Configuration for the training data sampler.
"""
struct SamplerConfig
    n_stations_range::UnitRange{Int}
    sr::Int
    freq_range::Tuple{Float32,Float32}
    wpm_range::Tuple{Float32,Float32}
    jitter_range::Tuple{Float32,Float32}
    amp_range::Tuple{Float32,Float32}
    noise_range::Tuple{Float32,Float32}
    spec::SpectrogramConfig
end

SamplerConfig(;
    n_stations_range = 1:3,
    sr               = 8000,
    freq_range       = (250f0, 750f0),
    wpm_range        = (15f0, 40f0),
    jitter_range     = (0.08f0, 0.25f0),
    amp_range        = (0.3f0, 1.0f0),
    noise_range      = (0.005f0, 0.08f0),
    spec             = SpectrogramConfig(),
) = SamplerConfig(n_stations_range, sr, freq_range, wpm_range,
                  jitter_range, amp_range, noise_range, spec)

"""
    generate_sample(cfg; rng, text_fn) → Sample

Produce one training sample: a random band scene → spectrogram + labels.
"""
function generate_sample(cfg::SamplerConfig;
                         rng::AbstractRNG = Random.default_rng(),
                         text_fn::Function = random_text)
    scene = random_band(rng;
        n_stations   = rand(rng, cfg.n_stations_range),
        sr           = cfg.sr,
        freq_range   = cfg.freq_range,
        wpm_range    = cfg.wpm_range,
        jitter_range = cfg.jitter_range,
        amp_range    = cfg.amp_range,
        noise_range  = cfg.noise_range,
        text_fn      = text_fn,
    )
    spec  = compute_spectrogram(scene.audio, scene.sr, cfg.spec)
    freqs = Float32[s.frequency for s in scene.stations]

    Sample(spec, scene.texts, freqs, length(scene.stations))
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Batch collation
# ═══════════════════════════════════════════════════════════════════════════════

"""
A padded batch of training samples, ready for the network.

Fields:
- `spectrogram`     : (freq_bins, batch, max_time)  — zero-padded
- `targets`         : (batch, max_stations, max_seq) — index-encoded, zero-padded
- `target_lengths`  : (batch, max_stations)
- `input_lengths`   : (batch,)
- `n_stations`      : (batch,)
- `frequencies`     : (batch, max_stations) — carrier Hz (0 for absent stations)
"""
struct Batch
    spectrogram::Array{Float32,3}
    targets::Array{Int,3}
    target_lengths::Matrix{Int}
    input_lengths::Vector{Int}
    n_stations::Vector{Int}
    frequencies::Matrix{Float32}
end

"""
    collate(samples) → Batch

Pad a vector of `Sample` into uniform-size tensors.
"""
function collate(samples::AbstractVector{Sample})
    B        = length(samples)
    max_ns   = maximum(s.n_stations for s in samples)
    max_time = maximum(size(s.spectrogram, 2) for s in samples)
    max_seq  = maximum(maximum(length(t) for t in s.texts) for s in samples)
    n_bins   = size(first(samples).spectrogram, 1)

    spec_batch = zeros(Float32, n_bins, B, max_time)
    tgt_batch  = zeros(Int, B, max_ns, max_seq)
    tgt_lens   = zeros(Int, B, max_ns)
    in_lens    = zeros(Int, B)
    ns_vec     = zeros(Int, B)
    freq_batch = zeros(Float32, B, max_ns)

    for (b, s) in enumerate(samples)
        T = size(s.spectrogram, 2)
        spec_batch[:, b, 1:T] .= s.spectrogram
        in_lens[b]  = T
        ns_vec[b]   = s.n_stations
        for k in 1:s.n_stations
            enc = encode_text(s.texts[k])
            L   = length(enc)
            tgt_batch[b, k, 1:L] .= enc
            tgt_lens[b, k] = L
            freq_batch[b, k] = s.frequencies[k]
        end
    end

    Batch(spec_batch, tgt_batch, tgt_lens, in_lens, ns_vec, freq_batch)
end

"""
    generate_batch(cfg, batch_size; rng, text_fn, parallel) → Batch

Generate and collate a full training batch. If `parallel=true` (default when
`Threads.nthreads() > 1`) and Julia was started with multiple threads, samples
are generated in parallel to keep the GPU data pipeline fed.
"""
function generate_batch(cfg::SamplerConfig, batch_size::Int;
                        rng::AbstractRNG = Random.default_rng(),
                        text_fn::Function = random_text,
                        parallel::Bool = Threads.nthreads() > 1)
    if parallel && Threads.nthreads() > 1
        base_seed = rand(rng, UInt)
        samples = Vector{Sample}(undef, batch_size)
        Threads.@threads for i in 1:batch_size
            rng_i = MersenneTwister(hash(base_seed, UInt(i)))
            @inbounds samples[i] = generate_sample(cfg; rng=rng_i, text_fn)
        end
        collate(samples)
    else
        samples = [generate_sample(cfg; rng, text_fn) for _ in 1:batch_size]
        collate(samples)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Real-time streaming: BandStream
# ═══════════════════════════════════════════════════════════════════════════════

"""
Per-station state in a `BandStream`.
"""
mutable struct StationStream
    station::Station
    audio::Vector{Float32}       # accumulated audio buffer
    messages::Vector{String}     # transmitted messages (in order)
    message_ends::Vector{Int}    # sample index where each message ends
end

"""
    BandStream(stations; sr, noise_σ, spec, rng, text_fn)

A stateful multi-station stream.  Call `next_chunk!` to get successive
spectrogram windows + per-station text labels — simulates listening to a
live band.

Each station continuously transmits random messages.  Audio is generated
lazily (only when more samples are needed).
"""
mutable struct BandStream
    streams::Vector{StationStream}
    noise_σ::Float32
    sr::Int
    spec::SpectrogramConfig
    position::Int                # current read position (sample index)
    rng::AbstractRNG
    text_fn::Function
end

function BandStream(stations::AbstractVector{Station};
                    sr::Int               = 8000,
                    noise_σ::Float32      = 0.02f0,
                    spec::SpectrogramConfig = SpectrogramConfig(),
                    rng::AbstractRNG      = Random.default_rng(),
                    text_fn::Function     = random_text)
    streams = [StationStream(s, Float32[], String[], Int[]) for s in stations]
    BandStream(streams, noise_σ, sr, spec, 0, rng, text_fn)
end

"""
    next_chunk!(stream, n_samples) → (spectrogram, texts)

Advance the stream by `n_samples` and return the spectrogram + per-station
text labels for the chunk.

`texts[i]` is the message station `i` was transmitting during this chunk.
"""
function next_chunk!(stream::BandStream, n_samples::Int)
    needed = stream.position + n_samples

    # Ensure each station has enough audio
    for ss in stream.streams
        while length(ss.audio) < needed
            msg   = stream.text_fn(stream.rng)
            chunk = synthesize(ss.station, msg, stream.sr, stream.rng)
            append!(ss.audio, chunk)
            push!(ss.messages, msg)
            push!(ss.message_ends, length(ss.audio))
        end
    end

    # Mix the requested window
    mixed = zeros(Float32, n_samples)
    start = stream.position + 1
    stop  = stream.position + n_samples

    for ss in stream.streams
        @views mixed .+= ss.audio[start:stop]
    end

    if stream.noise_σ > 0f0
        mixed .+= stream.noise_σ .* randn(stream.rng, Float32, n_samples)
    end
    clamp!(mixed, -1f0, 1f0)

    spec = compute_spectrogram(mixed, stream.sr, stream.spec)

    # Find current text per station (the message active at midpoint of chunk)
    mid = stream.position + n_samples ÷ 2
    texts = [active_message_at_position(ss, mid) for ss in stream.streams]

    stream.position = stop
    (spec, texts)
end

"""Find which message was active at sample index `pos`."""
function active_message_at_position(ss::StationStream, pos::Int)
    for (i, e) in enumerate(ss.message_ends)
        prev = i == 1 ? 0 : ss.message_ends[i-1]
        if prev < pos ≤ e
            return ss.messages[i]
        end
    end
    isempty(ss.messages) ? "" : last(ss.messages)
end

"""Reset stream to the beginning (buffers are kept)."""
function reset!(stream::BandStream)
    stream.position = 0
    stream
end
