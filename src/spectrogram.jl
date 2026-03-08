"""
    spectrogram.jl — Short-time FFT → power spectrogram in a frequency band.

Produces a (freq_bins × time_frames) matrix: one column per hop, only the
bins in [freq_lo, freq_hi] Hz.  This is the input representation for the
decoder network.
"""

# ─── Configuration ───────────────────────────────────────────────────────────

struct SpectrogramConfig
    nfft::Int
    hop::Int
    freq_lo::Float32
    freq_hi::Float32
end

SpectrogramConfig(; nfft=512, hop=128, freq_lo=200f0, freq_hi=800f0) =
    SpectrogramConfig(nfft, hop, freq_lo, freq_hi)

"""Number of frequency bins in the configured band for sample rate `sr`."""
function num_bins(cfg::SpectrogramConfig, sr::Int)
    lo = 1 + floor(Int, cfg.freq_lo * cfg.nfft / sr)
    hi = 1 + floor(Int, cfg.freq_hi * cfg.nfft / sr)
    hi - lo + 1
end

"""Number of time frames for `n_samples` audio at given config."""
num_frames(cfg::SpectrogramConfig, n_samples::Int) =
    max(1, (n_samples - cfg.nfft) ÷ cfg.hop + 1)

# ─── Window ──────────────────────────────────────────────────────────────────

"""Hann window of length `n`."""
hann(n::Int) = Float32[0.5f0 * (1f0 - cospi(2f0 * k / (n - 1))) for k in 0:n-1]

# ─── Core ────────────────────────────────────────────────────────────────────

"""
    compute_spectrogram(audio, sr, cfg) → Matrix{Float32}

Compute power spectrogram in the [freq_lo, freq_hi] band.
Returns a `(num_bins × num_frames)` matrix — ready to feed to the network.
"""
function compute_spectrogram(audio::AbstractVector{<:Real}, sr::Int,
                             cfg::SpectrogramConfig)
    n = length(audio)
    n_frames = num_frames(cfg, n)

    lo_bin = 1 + floor(Int, cfg.freq_lo * cfg.nfft / sr)
    hi_bin = 1 + floor(Int, cfg.freq_hi * cfg.nfft / sr)
    n_bins = hi_bin - lo_bin + 1

    spec = Matrix{Float32}(undef, n_bins, n_frames)
    win  = hann(cfg.nfft)
    buf  = Vector{Float32}(undef, cfg.nfft)

    @inbounds for f in 1:n_frames
        start = (f - 1) * cfg.hop + 1
        stop  = min(start + cfg.nfft - 1, n)
        len   = stop - start + 1

        # Windowed frame (zero-pad if at boundary)
        for i in 1:len
            buf[i] = Float32(audio[start + i - 1]) * win[i]
        end
        for i in len+1:cfg.nfft
            buf[i] = 0f0
        end

        power = abs2.(rfft(buf))
        spec[:, f] .= @view power[lo_bin:hi_bin]
    end

    spec
end
