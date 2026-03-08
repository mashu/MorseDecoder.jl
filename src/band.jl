"""
    band.jl — Multi-station mixing: overlay N stations into one audio signal.

Simulates a busy band segment where multiple CW stations transmit
simultaneously at different carrier frequencies.  The mixed audio + station
metadata forms the basis for training data.
"""

# ─── Band scene ──────────────────────────────────────────────────────────────

"""
Result of mixing multiple stations.

Fields:
- `audio`    : mixed audio waveform (sum of all stations + noise)
- `stations` : per-station parameters
- `texts`    : per-station transmitted text
- `sr`       : sample rate (Hz)
"""
struct BandScene
    audio::Vector{Float32}
    stations::Vector{Station}
    texts::Vector{String}
    sr::Int
end

# ─── Mixing ──────────────────────────────────────────────────────────────────

"""
    mix_stations(stations, texts, sr, rng; noise_σ) → BandScene

Synthesize each station independently, sum into a single waveform, add noise.
"""
function mix_stations(stations::AbstractVector{Station},
                      texts::AbstractVector{<:AbstractString},
                      sr::Int, rng::AbstractRNG;
                      noise_σ::Float32 = 0.02f0)
    @assert length(stations) == length(texts)

    # Generate per-station audio
    audios = [synthesize(s, t, sr, rng) for (s, t) in zip(stations, texts)]

    # Mix into common-length buffer
    max_len = maximum(length, audios)
    mixed   = zeros(Float32, max_len)
    for a in audios
        @views mixed[1:length(a)] .+= a
    end

    # Additive white Gaussian noise
    if noise_σ > 0f0
        mixed .+= noise_σ .* randn(rng, Float32, max_len)
    end

    clamp!(mixed, -1f0, 1f0)
    BandScene(mixed, collect(stations), collect(texts), sr)
end

# ─── Random band generation ─────────────────────────────────────────────────

"""
    random_band(rng; n_stations, sr, kw...) → BandScene

Generate a random multi-station band scene — the primary data source for
training.  Station frequencies are spread across the band with a minimum
separation so they remain distinguishable in the spectrogram.
"""
function random_band(rng::AbstractRNG;
                     n_stations::Int       = rand(rng, 1:4),
                     sr::Int               = 8000,
                     freq_range::Tuple     = (250f0, 750f0),
                     wpm_range::Tuple      = (15f0, 40f0),
                     jitter_range::Tuple   = (0.08f0, 0.25f0),
                     amp_range::Tuple      = (0.3f0, 1.0f0),
                     noise_range::Tuple    = (0.005f0, 0.08f0),
                     text_fn::Function     = random_text)

    freqs    = _spread_frequencies(rng, n_stations, freq_range)
    stations = [Station(;
        frequency = freqs[i],
        wpm       = _uniform(rng, wpm_range...),
        jitter    = _uniform(rng, jitter_range...),
        amplitude = _uniform(rng, amp_range...),
    ) for i in 1:n_stations]

    texts   = [text_fn(rng) for _ in 1:n_stations]
    noise_σ = _uniform(rng, noise_range...)

    mix_stations(stations, texts, sr, rng; noise_σ)
end

# ─── Helpers ─────────────────────────────────────────────────────────────────

"""Sample N frequencies with minimum separation within a range."""
function _spread_frequencies(rng::AbstractRNG, n::Int,
                             range::Tuple{Float32,Float32})
    lo, hi = range
    n == 0 && return Float32[]
    n == 1 && return Float32[_uniform(rng, lo, hi)]

    # Divide band into n equal slots, pick one freq per slot
    slot = (hi - lo) / n
    freqs = Float32[lo + (i - 1) * slot + rand(rng, Float32) * slot * 0.8f0
                    for i in 1:n]
    clamp.(freqs, lo, hi)
end

"""Uniform random Float32 in [lo, hi]."""
_uniform(rng::AbstractRNG, lo::Real, hi::Real) =
    Float32(lo) + rand(rng, Float32) * Float32(hi - lo)
