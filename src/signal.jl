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

# ─── Envelope smoothing ───────────────────────────────────────────────────────

"""
    smooth_envelope(env, sr; ramp_ms) → Vector{Float32}

Apply attack/release ramps to a 0/1 keying envelope to avoid clicks.
Uses Hann-style ramps (sin²) so the envelope and its derivative are continuous
at boundaries.  `ramp_ms`: ramp duration in ms (default 5); use 4–8 ms for
clearly smooth edges.
"""
function smooth_envelope(env::Vector{Float32}, sr::Int; ramp_ms::Real=5f0)
    ramp_samples = max(1, round(Int, sr * Float32(ramp_ms) / 1000f0))
    out = copy(env)
    n = length(env)
    i = 1
    while i <= n
        if out[i] >= 0.5f0  # start of 1-run
            j = i
            while j <= n && out[j] >= 0.5f0
                j += 1
            end
            j -= 1
            run_len = j - i + 1
            # Use up to ramp_samples per edge; for short runs use one smooth hump
            half = run_len ÷ 2
            attack_len = min(ramp_samples, half)
            release_len = min(ramp_samples, half)
            # Ensure at least 1 sample ramp so we never leave a hard step
            attack_len = max(1, attack_len)
            release_len = max(1, release_len)
            if attack_len + release_len > run_len
                attack_len = run_len ÷ 2
                release_len = run_len - attack_len
            end
            # Hann (sin²) ramp 0→1: smooth and zero derivative at both ends
            for k in 0:(attack_len - 1)
                x = Float32(k) / attack_len
                @inbounds out[i + k] = sin(Float32(π) * x * 0.5f0)^2
            end
            # Release: 1 → 0 so that out[j] = 0 and out[j-release_len+1] = 1
            for k in 0:(release_len - 1)
                x = release_len <= 1 ? 0f0 : Float32(k) / (release_len - 1)
                @inbounds out[j - k] = sin(Float32(π) * x * 0.5f0)^2
            end
            i = j + 1
        else
            i += 1
        end
    end
    out
end

# ─── Synthesis ───────────────────────────────────────────────────────────────

"""
    synthesize(station, text, sr, rng; ramp_ms) → Vector{Float32}

Generate audio for one station transmitting `text`.

Pipeline:  text → keying envelope → smooth envelope (attack/release) →
carrier modulation → amplitude scaling.  `ramp_ms` (default 5) sets the
envelope ramp length in ms to reduce keying clicks.
"""
function synthesize(station::Station, text::AbstractString, sr::Int,
                    rng::AbstractRNG; ramp_ms::Real=5f0)
    env = keying_envelope(text, station.wpm, sr, station.jitter, rng)
    env = smooth_envelope(env, sr; ramp_ms)
    n   = length(env)
    ω   = 2f0 * Float32(π) * station.frequency / sr

    audio = Vector{Float32}(undef, n)
    @inbounds for i in 1:n
        audio[i] = station.amplitude * env[i] * sin(ω * (i - 1))
    end
    audio
end
