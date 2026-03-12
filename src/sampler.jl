"""
    sampler.jl — Training samples via MorseSimulator.jl.

Uses MorseSimulator's DatasetConfig with 200–900 Hz mel band, ~10 Hz frequency
resolution (to separate signals), and sufficient time resolution for dots/dashes
up to 50 WPM. Produces (spectrogram, token_ids) pairs; token_ids use vocab.jl
(<START>, <END>, [S1]..[S6], [TS], [TE], chars).
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
- `max_frames`     : cap spectrogram time dimension for GPU (nothing = no cap).
"""
struct SamplerConfig
    dataset_config::DatasetConfig
    max_frames::Union{Int,Nothing}
end

# 200–900 Hz only (Morse CW). ~10 Hz freq resolution → fft_size ≥ sr/10 (44100/10=4410 → 4096).
# 50 WPM: dit ≈ 24 ms; ≥2–3 frames per dit → hop ≤ 8–12 ms; hop 256 @ 44.1 kHz ≈ 5.8 ms.
function SamplerConfig(;
    path = DirectPath(),
    sample_rate::Int = 44100,
    fft_size::Int = 4096,      # ~10.8 Hz bin at 44.1 kHz
    hop_size::Int = 256,       # ~5.8 ms/frame, good for 50 WPM
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

One training sample from MorseSimulator (spectrogram + label as token_ids).
"""
function generate_sample(cfg::SamplerConfig; rng::AbstractRNG = Random.default_rng())
    ds = sim_generate_sample(rng, cfg.dataset_config)
    spec = Float32.(ds.mel_spectrogram)
    if cfg.max_frames !== nothing && size(spec, 2) > cfg.max_frames
        spec = spec[:, 1:cfg.max_frames]
    end
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
