"""
Type hierarchy for composable Morse signal simulation.

The simulator is built from three orthogonal axes, each governed by an
abstract type:

  • `Modulation`   – how the keying envelope becomes an RF/audio signal
  • `NoiseModel`   – additive channel impairments
  • `KeyingStyle`  – timing jitter model for dit/dah lengths

Concrete subtypes carry the parameters for each axis, and `MorseSignalConfig`
bundles one of each together with global settings (WPM, amplitude, sample rate).
Multiple dispatch on the three axes gives a clean extension point: to add
Rayleigh fading, just define a new `FadingNoise <: NoiseModel` and a method
for `apply_noise`.
"""

# ── abstract roots ────────────────────────────────────────────────────────────

"""Carrier modulation applied to the on/off keying envelope."""
abstract type Modulation end

"""Additive noise / impairment model."""
abstract type NoiseModel end

"""Timing model for key-down / key-up durations."""
abstract type KeyingStyle end

# ── concrete modulation types ─────────────────────────────────────────────────

"""Pure sine-wave modulation at a fixed frequency (Hz)."""
struct SineModulation{T<:AbstractFloat} <: Modulation
    frequency::T
end
SineModulation(f::Real) = SineModulation(Float32(f))

"""Carrier frequency in Hz (for manifest); 0 for non-sine modulation."""
carrier_frequency(m::SineModulation) = m.frequency
carrier_frequency(::Modulation) = 0f0

# ── concrete noise types ──────────────────────────────────────────────────────

"""Additive white Gaussian noise characterised by a power parameter."""
struct WhiteGaussianNoise{T<:AbstractFloat} <: NoiseModel
    power::T
end
WhiteGaussianNoise(p::Real) = WhiteGaussianNoise(Float32(p))

"""No noise (clean signal)."""
struct NoNoise <: NoiseModel end

"""Noise power (for manifest); 0 for no-noise."""
noise_power(n::WhiteGaussianNoise) = n.power
noise_power(::NoiseModel) = 0f0

# ── concrete keying styles ────────────────────────────────────────────────────

"""
Human-like keying: each dit/dah length is drawn from
`Normal(nominal, nominal * σ)` then clipped to `[0.5, 2.0] × nominal`.
"""
struct HumanKeying{T<:AbstractFloat} <: KeyingStyle
    timing_sigma::T
end
HumanKeying() = HumanKeying(Float32(0.2))

"""Perfect machine keying — no timing jitter at all."""
struct PerfectKeying <: KeyingStyle end

"""Base keying for element-level timing (identity for non-drifting)."""
base_keying(k::KeyingStyle) = k

"""Max WPM drift (for manifest); 0 for non-drifting keying."""
max_drift(::KeyingStyle) = 0f0

# ── top-level configuration ───────────────────────────────────────────────────

"""
Complete specification of a synthetic Morse signal.

# Fields
- `modulation`      : carrier modulation
- `noise`           : channel noise model
- `keying`          : timing jitter model
- `wpm`             : words per minute (PARIS standard)
- `amplitude`       : output amplitude as percentage (clipped to [-1, 1])
- `sample_rate`     : samples per second (Hz)
- `chunk_duration_s`: chunk length in seconds (one FFT token per chunk)
- `nfft`            : FFT length (zero-pad for finer frequency resolution in 100–900 Hz)
"""
struct MorseSignalConfig{M<:Modulation, N<:NoiseModel, K<:KeyingStyle, T<:AbstractFloat}
    modulation::M
    noise::N
    keying::K
    wpm::T
    amplitude::T
    sample_rate::Int
    chunk_duration_s::Float64
    nfft::Int
end

function MorseSignalConfig(;
    modulation::Modulation = SineModulation(500f0),
    noise::NoiseModel      = WhiteGaussianNoise(1f0),
    keying::KeyingStyle    = HumanKeying(),
    wpm::Real              = 20f0,
    amplitude::Real         = 100f0,
    sample_rate::Int       = 2000,
    chunk_duration_s::Real = 0.04,
    nfft::Int              = 512,
)
    T = Float32
    MorseSignalConfig(modulation, noise, keying, T(wpm), T(amplitude), sample_rate,
                      Float64(chunk_duration_s), nfft)
end

"""
Result of a Morse signal generation run.  Carries the raw audio, the
spectrogram fed to the network, and the ground-truth text.
"""
struct MorseSignal{A<:AbstractVector, S<:AbstractMatrix}
    audio::A
    spectrogram::S
    text::String
    config::MorseSignalConfig
end
