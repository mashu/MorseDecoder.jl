"""
    signal.jl — A single Morse station: carrier synthesis from text.

A `Station` is a transmitter with a fixed carrier frequency, speed, timing
jitter, and amplitude.  `synthesize` produces the audio waveform for a given
text string.
"""

# ─── Station ─────────────────────────────────────────────────────────────────

struct Station
    frequency::Float32     # carrier Hz (200–800 typical)
    wpm::Float32           # words per minute
    jitter::Float32        # timing jitter σ  (0 = perfect, 0.15–0.25 = human)
    amplitude::Float32     # signal strength   (0–1)
end

Station(; frequency, wpm, jitter=0.15f0, amplitude=0.8f0) =
    Station(Float32(frequency), Float32(wpm), Float32(jitter), Float32(amplitude))

# ─── Synthesis ───────────────────────────────────────────────────────────────

"""
    synthesize(station, text, sr, rng) → Vector{Float32}

Generate audio for one station transmitting `text`.

Pipeline:  text → keying envelope → carrier modulation → amplitude scaling.
"""
function synthesize(station::Station, text::AbstractString, sr::Int,
                    rng::AbstractRNG)
    env = keying_envelope(text, station.wpm, sr, station.jitter, rng)
    n   = length(env)
    ω   = 2f0 * Float32(π) * station.frequency / sr

    audio = Vector{Float32}(undef, n)
    @inbounds for i in 1:n
        audio[i] = station.amplitude * env[i] * sin(ω * (i - 1))
    end
    audio
end
