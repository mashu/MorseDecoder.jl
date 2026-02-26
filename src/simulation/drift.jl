"""
WPM drift: vary sending speed across word boundaries within a single
transmission, modelling real-world human CW behaviour.

Operators naturally speed up and slow down — especially in contests where
they'll blast the callsign fast but slow down for the exchange, or vice versa.

The `DriftingKeying` wrapper composes with any underlying `KeyingStyle` and
adds per-word WPM jitter.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  Drifting keying wrapper
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DriftingKeying{K<:KeyingStyle, T<:AbstractFloat}

Wraps an underlying keying style and adds per-word WPM drift.

# Fields
- `base_keying` : the underlying timing model (e.g. `HumanKeying`)
- `max_drift`   : maximum WPM deviation from nominal (e.g. 5.0 for ±5 WPM)
"""
struct DriftingKeying{K<:KeyingStyle, T<:AbstractFloat} <: KeyingStyle
    base_keying::K
    max_drift::T
end
DriftingKeying(k::KeyingStyle, drift::Real) = DriftingKeying(k, Float32(drift))
DriftingKeying(; max_drift::Real = 5f0) = DriftingKeying(HumanKeying(), Float32(max_drift))

base_keying(dk::DriftingKeying) = dk.base_keying
max_drift(dk::DriftingKeying) = dk.max_drift

# Forward dit/dah sampling to the underlying keying style
sample_dit(dk::DriftingKeying, nominal::Real, rng::AbstractRNG) =
    sample_dit(dk.base_keying, nominal, rng)

sample_dah(dk::DriftingKeying, nominal::Real, rng::AbstractRNG) =
    sample_dah(dk.base_keying, nominal, rng)

# ═══════════════════════════════════════════════════════════════════════════════
#  Per-word WPM drift generation
# ═══════════════════════════════════════════════════════════════════════════════

"""Per-word WPMs: use drift when keying is DriftingKeying, else constant."""
function word_wpms_for_exchange(dk::DriftingKeying, wpm::Real, n_words::Int, rng::AbstractRNG)
    n_words ≤ 1 && return fill(Float32(wpm), n_words)
    sample_word_wpms(rng, wpm, dk.max_drift, n_words)
end
word_wpms_for_exchange(::KeyingStyle, wpm::Real, n_words::Int, ::AbstractRNG) =
    fill(Float32(wpm), n_words)

"""
    sample_word_wpms(rng, base_wpm, max_drift, n_words) → Vector{Float32}

Generate per-word WPM values with smooth random-walk drift.
"""
function sample_word_wpms(rng::AbstractRNG, base_wpm::Real,
                          max_drift::Real, n_words::Int)
    n_words ≤ 0 && return Float32[]
    n_words == 1 && return Float32[base_wpm]

    # Random walk with mean reversion
    wpms = Vector{Float32}(undef, n_words)
    current = Float32(0)   # offset from base
    map(1:n_words) do i
        # Mean-revert: pull back toward centre
        step = randn(rng) * Float32(max_drift * 0.3) - current * Float32(0.2)
        current = clamp(current + step, -Float32(max_drift), Float32(max_drift))
        wpms[i] = Float32(base_wpm) + current
        nothing
    end
    clamp.(wpms, 10f0, 60f0)  # sane limits
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Generate signal from ContestExchange with per-word WPM drift
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generate_contest_signal(config, exchange; rng) → MorseSignal

Generate a Morse audio signal from a `ContestExchange`, with optional
per-word WPM drift.

When `config.keying` is a `DriftingKeying`, each word in `exchange.words`
is synthesised at a slightly different WPM.  The word envelopes are then
concatenated with inter-word gaps, modulated, and noised as usual.
"""
function generate_contest_signal(config::MorseSignalConfig,
                                 exchange::ContestExchange;
                                 rng::AbstractRNG = Random.default_rng())

    (; modulation, noise, keying, wpm, amplitude, sample_rate, chunk_duration_s, nfft) = config

    n_words = length(exchange.words)
    word_wpms = word_wpms_for_exchange(keying, wpm, n_words, rng)
    actual_keying = base_keying(keying)

    # Build envelope: each word at its own WPM, with inter-word gaps
    segments = Vector{Vector{Float32}}()

    # Leading silence
    nominal_lead = dit_samples(wpm, sample_rate)
    push!(segments, zeros(Float32, 5 * sample_dit(actual_keying, nominal_lead, rng)))

    word_envelopes = map(enumerate(exchange.words)) do (wi, word)
        w_wpm    = word_wpms[wi]
        nominal  = dit_samples(w_wpm, sample_rate)
        chars    = collect(word)

        # Encode each character
        char_envs = map(chars) do c
            if c == ' '
                # Should not happen (words are split on spaces) but handle gracefully
                zeros(Float32, 7 * sample_dit(actual_keying, nominal, rng))
            else
                encode_character(c, actual_keying, nominal, rng)
            end
        end

        reduce(vcat, char_envs)
    end

    # Interleave word envelopes with inter-word gaps (7 dit lengths)
    map(enumerate(word_envelopes)) do (i, env)
        push!(segments, env)
        if i < n_words
            # Inter-word gap: 7 dits at average of adjacent word speeds
            gap_wpm = (word_wpms[i] + word_wpms[min(i + 1, n_words)]) / 2
            gap_nominal = dit_samples(gap_wpm, sample_rate)
            push!(segments, zeros(Float32, 7 * sample_dit(actual_keying, gap_nominal, rng)))
        end
        nothing
    end

    # Trailing silence
    push!(segments, zeros(Float32, 5 * sample_dit(actual_keying, nominal_lead, rng)))

    envelope = reduce(vcat, segments)
    modulated = modulate(modulation, envelope, sample_rate)
    noisy = apply_noise(noise, modulated, sample_rate, rng)
    audio = scale_and_clip(noisy, amplitude)
    fft_cfg = MorseFFTConfig(sample_rate, chunk_duration_s, nfft)
    spec = audio_to_tokens(audio, fft_cfg)

    MorseSignal(audio, spec, exchange.text, config)
end

"""
    generate_contest_signal(; rng, kwargs...) → (MorseSignal, ContestExchange)

Convenience: random contest, random config with drift, all randomised.
Returns both the signal and the exchange for metadata.
"""
function generate_contest_signal(; rng::AbstractRNG = Random.default_rng(),
                                   sample_rate::Int = 2000,
                                   pitch_range  = 300:900,
                                   wpm_range    = 18:45,
                                   noise_range  = 0:200,
                                   amp_range    = 20:150,
                                   max_drift    = 5f0,
                                   contest::Union{AbstractContest, Nothing} = nothing)

    cfg = MorseSignalConfig(;
        modulation  = SineModulation(Float32(rand(rng, pitch_range))),
        noise       = WhiteGaussianNoise(Float32(rand(rng, noise_range))),
        keying      = DriftingKeying(HumanKeying(), max_drift),
        wpm         = Float32(rand(rng, wpm_range)),
        amplitude   = Float32(rand(rng, amp_range)),
        sample_rate,
    )

    exch = isnothing(contest) ? random_exchange(rng) : generate_exchange(contest, rng)
    sig  = generate_contest_signal(cfg, exch; rng)
    (sig, exch)
end
