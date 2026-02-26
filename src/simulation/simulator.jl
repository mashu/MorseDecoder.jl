"""
Morse signal simulator built on multiple dispatch.

Key design: each of the three simulation axes (modulation, noise, keying)
dispatches independently, so mixing and matching is trivial.  The top-level
entry point `generate_signal` assembles them.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  Keying – produce element durations (in samples)
# ═══════════════════════════════════════════════════════════════════════════════

"""Nominal dit length in samples for a given WPM and sample rate."""
dit_samples(wpm::Real, sr::Int) = (60 / wpm) / 50 * sr   # PARIS = 50 dits

"""Sample a single dit duration, optionally jittered."""
function sample_dit(keying::HumanKeying, nominal::Real, rng::AbstractRNG)
    σ = keying.timing_sigma
    scale = clamp(randn(rng) * σ + one(σ), Float32(0.5), Float32(2.0))
    round(Int, nominal * scale)
end

"""Sample a single dah duration (≈3× dit), optionally jittered."""
function sample_dah(keying::HumanKeying, nominal::Real, rng::AbstractRNG)
    σ = keying.timing_sigma
    scale = clamp(randn(rng) * σ + one(σ), Float32(0.5), Float32(2.0))
    round(Int, 3 * nominal * scale)
end

sample_dit(::PerfectKeying, nominal::Real, ::AbstractRNG) = round(Int, nominal)
sample_dah(::PerfectKeying, nominal::Real, ::AbstractRNG) = round(Int, 3 * nominal)

# ═══════════════════════════════════════════════════════════════════════════════
#  Envelope generation – build the on/off keying waveform
# ═══════════════════════════════════════════════════════════════════════════════

"""
    encode_character(char, keying, nominal_dit, rng) → Vector{Float32}

Produce the keying envelope for a single Morse character (one letter/digit).
Uses `map` + `reduce(vcat, …)` — no explicit loops or indexing.
"""
function encode_character(c::Char, keying::KeyingStyle, nominal::Real, rng::AbstractRNG)
    morse = MORSE_TABLE[c]
    symbols = collect(morse)

    # Each Morse element → [on-segment, inter-element gap]
    element_segments = map(symbols) do sym
        on_len  = sym == '.' ? sample_dit(keying, nominal, rng) :
                                sample_dah(keying, nominal, rng)
        gap_len = sample_dit(keying, nominal, rng)
        vcat(ones(Float32, on_len), zeros(Float32, gap_len))
    end

    # Concatenate all element segments, then add inter-character gap (2 dits extra)
    char_envelope = reduce(vcat, element_segments)
    inter_char    = zeros(Float32, 2 * sample_dit(keying, nominal, rng))
    vcat(char_envelope, inter_char)
end

"""
    build_envelope(text, keying, wpm, sample_rate, rng) → Vector{Float32}

Build the complete keying envelope for `text`.
Spaces map to 7-dit silences; other characters go through `encode_character`.
"""
function build_envelope(text::AbstractString, keying::KeyingStyle,
                        wpm::Real, sr::Int, rng::AbstractRNG)
    nominal = dit_samples(wpm, sr)
    chars   = collect(text)

    # Leading silence
    segments = Vector{Vector{Float32}}()
    push!(segments, zeros(Float32, 5 * sample_dit(keying, nominal, rng)))

    # Map each character to its envelope segment
    char_segments = map(chars) do c
        if c == ' '
            zeros(Float32, 7 * sample_dit(keying, nominal, rng))
        else
            encode_character(c, keying, nominal, rng)
        end
    end
    append!(segments, char_segments)

    # Trailing silence
    push!(segments, zeros(Float32, 5 * sample_dit(keying, nominal, rng)))

    reduce(vcat, segments)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Modulation – multiply envelope by a carrier
# ═══════════════════════════════════════════════════════════════════════════════

"""
    modulate(mod, envelope, sample_rate) → Vector{Float32}

Apply carrier modulation to the keying envelope.
"""
function modulate(mod::SineModulation, envelope::AbstractVector{Float32}, sr::Int)
    n    = length(envelope)
    t    = Float32.(range(0, step = 1 / sr, length = n))
    sine = sin.(2f0 * Float32(π) * mod.frequency .* t)
    sine .* envelope
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Noise – add channel impairments
# ═══════════════════════════════════════════════════════════════════════════════

"""
    apply_noise(noise, signal, sample_rate, rng) → Vector{Float32}

Add noise to a modulated signal.
"""
function apply_noise(noise::WhiteGaussianNoise, sig::AbstractVector{Float32},
                     sr::Int, rng::AbstractRNG)
    power_linear = 1f-6 * noise.power * sr / 2
    σ = sqrt(power_linear)
    n = randn(rng, Float32, length(sig))
    0.5f0 .* sig .+ σ .* n
end

apply_noise(::NoNoise, sig::AbstractVector{Float32}, ::Int, ::AbstractRNG) = 0.5f0 .* sig

# ═══════════════════════════════════════════════════════════════════════════════
#  Amplitude / clipping
# ═══════════════════════════════════════════════════════════════════════════════

"""Scale and clip a signal to ±1."""
function scale_and_clip(sig::AbstractVector{Float32}, amplitude::Real)
    clamp.(sig .* (amplitude / 100f0), -1f0, 1f0)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Random text generation
# ═══════════════════════════════════════════════════════════════════════════════

"""
Generate a random Morse-compatible string of length `n`.
First and last characters are guaranteed non-space.
"""
function random_morse_text(rng::AbstractRNG, n::Int)
    chars      = collect(ALPHABET)
    non_space  = filter(!isequal(' '), chars)
    first_char = non_space[rand(rng, 1:length(non_space))]
    last_char  = non_space[rand(rng, 1:length(non_space))]
    middle     = map(_ -> chars[rand(rng, 1:length(chars))], 1:max(0, n - 2))
    String(vcat(first_char, middle, last_char))
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Random configuration generation (for diverse training data)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    random_config(rng) → MorseSignalConfig

Draw a random signal configuration similar to the original nn-morse trainer.
"""
function random_config(rng::AbstractRNG;
                       sample_rate::Int = 2000,
                       pitch_range  = 100:950,
                       wpm_range    = 10:40,
                       noise_range  = 0:200,
                       amp_range    = 10:150,
                       chunk_duration_s::Real = 0.04,
                       nfft::Int = 512,
                       kwargs...)
    MorseSignalConfig(;
        modulation = SineModulation(Float32(rand(rng, pitch_range))),
        noise = WhiteGaussianNoise(Float32(rand(rng, noise_range))),
        keying = HumanKeying(),
        wpm = Float32(rand(rng, wpm_range)),
        amplitude = Float32(rand(rng, amp_range)),
        sample_rate,
        chunk_duration_s,
        nfft,
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Top-level entry point
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generate_signal(config, text; rng) → MorseSignal

Synthesise a complete Morse signal from `config` and `text`.
Returns a `MorseSignal` containing the raw audio waveform, its spectrogram,
the ground-truth text string, and the configuration used.
"""
function generate_signal(config::MorseSignalConfig, text::AbstractString;
                         rng::AbstractRNG = Random.default_rng())
    (; modulation, noise, keying, wpm, amplitude, sample_rate, chunk_duration_s, nfft) = config

    envelope = build_envelope(text, keying, wpm, sample_rate, rng)
    modulated = modulate(modulation, envelope, sample_rate)
    noisy = apply_noise(noise, modulated, sample_rate, rng)
    audio = scale_and_clip(noisy, amplitude)
    fft_cfg = MorseFFTConfig(sample_rate, chunk_duration_s, nfft)
    spec = audio_to_tokens(audio, fft_cfg)

    MorseSignal(audio, spec, text, config)
end

"""
    generate_signal(; text_length=10, rng, kw...) → MorseSignal

Convenience: generate a random configuration *and* random text.
"""
function generate_signal(; text_length::Int = 10,
                           rng::AbstractRNG = Random.default_rng(), kwargs...)
    cfg  = random_config(rng; kwargs...)
    text = random_morse_text(rng, text_length)
    generate_signal(cfg, text; rng)
end
