"""
Chunk audio → FFT each chunk → one token per chunk (power in 100–900 Hz).
Single responsibility; parametric types; dispatch only.
"""

# ── Config ───────────────────────────────────────────────────────────────────

struct MorseFFTConfig{I<:Integer, F<:AbstractFloat}
    sr::I
    chunk_len::I
    nfft::I
    f_lo::F
    f_hi::F
end

MorseFFTConfig(sr::Integer, chunk_len::Integer, nfft::Integer; f_lo::Real = 100, f_hi::Real = 900) =
    MorseFFTConfig(sr, chunk_len, nfft, Float64(f_lo), Float64(f_hi))

MorseFFTConfig(sr::Integer, chunk_duration_s::Real, nfft::Integer; f_lo::Real = 100, f_hi::Real = 900) =
    MorseFFTConfig(sr, round(Int, Float64(chunk_duration_s) * sr), nfft; f_lo, f_hi)

# ── Bin range (dispatch) ─────────────────────────────────────────────────────

"""1-based rfft bin index for frequency f_hz."""
bin_index(sr::Integer, nfft::Integer, f_hz::Real) = 1 + floor(Int, Float64(f_hz) * nfft / sr)

"""Range of bin indices covering [f_lo, f_hi] Hz."""
bin_range(sr::Integer, nfft::Integer, f_lo::Real, f_hi::Real) =
    bin_index(sr, nfft, f_lo):bin_index(sr, nfft, f_hi)

bin_range(cfg::MorseFFTConfig) = bin_range(cfg.sr, cfg.nfft, cfg.f_lo, cfg.f_hi)

"""Number of frequency bins in the configured band."""
n_bins(cfg::MorseFFTConfig) = length(bin_range(cfg))

# ── Chunks (single responsibility) ───────────────────────────────────────────

"""Zero-pad or truncate to length len."""
pad_to(x::AbstractVector{<:AbstractFloat}, len::Integer) =
    vcat(Float32.(x), zeros(Float32, max(0, len - length(x))))[1:len]

"""Non-overlapping chunks; last chunk zero-padded to chunk_len."""
chunks(audio::AbstractVector{<:AbstractFloat}, chunk_len::Integer) =
    [pad_to(view(audio, (k - 1) * chunk_len + 1:min(k * chunk_len, length(audio))), chunk_len)
     for k in 1:cld(length(audio), chunk_len)]

# ── FFT building blocks ──────────────────────────────────────────────────────

"""Power spectrum of one chunk (length nfft÷2+1)."""
power_spectrum(samples::AbstractVector{<:AbstractFloat}, nfft::Integer) =
    abs2.(rfft(pad_to(samples, nfft)))

"""Slice spectrum to band [f_lo, f_hi] (one token)."""
band(power::AbstractVector{<:AbstractFloat}, r::UnitRange{<:Integer}) = power[r]

# ── Chunk → token (dispatch) ────────────────────────────────────────────────

"""One chunk → one token (power in cfg band)."""
chunk_to_token(chunk::AbstractVector{<:AbstractFloat}, cfg::MorseFFTConfig) =
    band(power_spectrum(chunk, cfg.nfft), bin_range(cfg))

# ── Top level ───────────────────────────────────────────────────────────────

"""Audio → matrix (n_bins × n_tokens); each column is one token."""
function audio_to_tokens(audio::AbstractVector{<:AbstractFloat}, cfg::MorseFFTConfig)
    cs = chunks(audio, cfg.chunk_len)
    nb = n_bins(cfg)
    out = Matrix{Float32}(undef, nb, length(cs))
    for (j, c) in enumerate(cs)
        out[:, j] = chunk_to_token(c, cfg)
    end
    out
end

# ── Backward-compat name ────────────────────────────────────────────────────

"""Alias for audio_to_tokens; (freq_bins × n_tokens)."""
compute_spectrogram(audio::AbstractVector{<:AbstractFloat}, sr::Integer, cfg::MorseFFTConfig) =
    audio_to_tokens(audio, cfg)

compute_spectrogram(audio::AbstractVector{<:AbstractFloat}, sr::Integer; chunk_duration::Real, nfft::Integer, f_lo::Real = 100, f_hi::Real = 900) =
    audio_to_tokens(audio, MorseFFTConfig(sr, chunk_duration, nfft; f_lo, f_hi))
