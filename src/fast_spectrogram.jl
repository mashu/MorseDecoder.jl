"""
    fast_spectrogram.jl — Direct spectrogram synthesis for CW signals.

Instead of generating audio sample-by-sample and computing an FFT, we exploit
the fact that a CW signal is a sinusoid at a known carrier frequency modulated
by a known on/off envelope.  The power spectrogram of such a signal is:

    S[b, t] ≈ (A · env(t))² · |W(b - b₀)|²

where W is the spectral response of the analysis window (Hann) and b₀ is the
carrier bin.  For multiple stations we sum individual spectrograms (valid when
stations are separated in frequency).  Noise is added in spectrogram domain.

This is ~10–30× faster than audio synthesis + STFT because:
  • No per-sample sin() calls (the carrier is implicit)
  • No FFT (the window spectrum is precomputed)
  • Envelope → frame-rate downsampling is cheap

For inference on real WAV files, use compute_spectrogram (STFT path).
"""

using FFTW: rfft

# ─── Precomputed Hann window power spectrum ─────────────────────────────────

"""Precomputed spectral kernel for a Hann window of a given nfft."""
struct HannKernelCache
    nfft::Int
    full_spectrum::Vector{Float32}
end

function HannKernelCache(nfft::Int)
    win = hann(nfft)
    spec = rfft(win)
    full = Float32[abs2(s) for s in spec]
    HannKernelCache(nfft, full)
end

const _hann_cache = Dict{Int, HannKernelCache}()
const _hann_cache_lock = ReentrantLock()

function get_hann_cache(nfft::Int)
    # Fast path (no lock)
    c = get(_hann_cache, nfft, nothing)
    c !== nothing && return c
    lock(_hann_cache_lock) do
        get!(_hann_cache, nfft) do
            HannKernelCache(nfft)
        end
    end
end

"""
Extract the window power kernel for a carrier at `center_bin`, mapped onto
the output frequency range [lo_bin, hi_bin]. Returns a Vector{Float32} of
length n_bins.
"""
function carrier_kernel(cache::HannKernelCache, lo_bin::Int, hi_bin::Int, center_bin::Int)
    n_bins = hi_bin - lo_bin + 1
    n_spec = length(cache.full_spectrum)
    kernel = Vector{Float32}(undef, n_bins)
    @inbounds for i in 1:n_bins
        delta = abs(lo_bin + i - 1 - center_bin)
        kernel[i] = delta < n_spec ? cache.full_spectrum[delta + 1] : 0f0
    end
    kernel
end

# ─── Envelope downsampled to frame rate ──────────────────────────────────────

"""
    envelope_to_frames(env_audio, nfft, hop, n_frames) → Vector{Float32}

Downsample a sample-rate envelope to spectrogram frame rate by averaging the
squared envelope in each analysis window. This matches what the STFT power
would produce for a slowly-varying envelope.
"""
function envelope_to_frames(env_audio::Vector{Float32}, nfft::Int, hop::Int, n_frames::Int)
    n_audio = length(env_audio)
    frames = Vector{Float32}(undef, n_frames)
    @inbounds for f in 1:n_frames
        lo = (f - 1) * hop + 1
        hi = min(lo + nfft - 1, n_audio)
        if lo > n_audio
            frames[f] = 0f0
        else
            s = 0f0
            for i in lo:hi
                s += env_audio[i] * env_audio[i]
            end
            frames[f] = s / nfft  # normalize by window length
        end
    end
    frames
end

# ─── Direct spectrogram synthesis ────────────────────────────────────────────

"""
    synthesize_spectrogram(stations, texts, sr, rng, cfg; noise_σ) → Matrix{Float32}

Directly synthesize the power spectrogram for a multi-station CW scene.
No audio generation, no FFT — uses analytical spectrogram of carrier + envelope.
Returns (n_bins, n_frames) matrix matching compute_spectrogram output format.
"""
function synthesize_spectrogram(
    stations::AbstractVector{Station},
    texts::AbstractVector{<:AbstractString},
    sr::Int,
    rng::AbstractRNG,
    cfg::SpectrogramConfig;
    noise_σ::Float32 = 0.02f0,
)
    @assert length(stations) == length(texts)

    nfft = cfg.nfft
    hop = cfg.hop
    lo_bin = 1 + floor(Int, cfg.freq_lo * nfft / sr)
    hi_bin = 1 + floor(Int, cfg.freq_hi * nfft / sr)
    n_bins = hi_bin - lo_bin + 1

    cache = get_hann_cache(nfft)

    # Build per-station smoothed envelopes, find max length for frame count
    envelopes = Vector{Vector{Float32}}(undef, length(stations))
    max_audio_len = 0
    for (idx, (station, text)) in enumerate(zip(stations, texts))
        env = keying_envelope(text, station.wpm, sr, station.jitter, rng)
        env = smooth_envelope(env, sr)
        envelopes[idx] = env
        max_audio_len = max(max_audio_len, length(env))
    end
    n_frames = num_frames(cfg, max_audio_len)

    spec = zeros(Float32, n_bins, n_frames)

    for (idx, station) in enumerate(stations)
        center_bin = 1 + round(Int, station.frequency * nfft / sr)
        kernel = carrier_kernel(cache, lo_bin, hi_bin, center_bin)
        env_frames = envelope_to_frames(envelopes[idx], nfft, hop, n_frames)
        amp2 = station.amplitude * station.amplitude

        @inbounds for f in 1:n_frames
            e = amp2 * env_frames[f]
            for b in 1:n_bins
                spec[b, f] += e * kernel[b]
            end
        end
    end

    # Noise in spectrogram domain: power spectral density of white noise ≈ σ² per bin
    if noise_σ > 0f0
        noise_var = noise_σ * noise_σ * nfft
        @inbounds for j in 1:n_frames
            for i in 1:n_bins
                spec[i, j] += noise_var * abs(randn(rng, Float32))
            end
        end
    end

    spec
end

# ─── Fast sample / batch generation ─────────────────────────────────────────

"""
    generate_sample_fast(cfg; rng, text_fn) → Sample

Like generate_sample but uses direct spectrogram synthesis (no audio, no FFT).
~10-30× faster.
"""
function generate_sample_fast(cfg::SamplerConfig;
                              rng::AbstractRNG = Random.default_rng(),
                              text_fn::Function = random_text)
    n_stations = rand(rng, cfg.n_stations_range)
    freqs = spread_frequencies(rng, n_stations, cfg.freq_range)
    stations = [Station(;
        frequency = freqs[i],
        wpm       = uniform_float(rng, cfg.wpm_range...),
        jitter    = uniform_float(rng, cfg.jitter_range...),
        amplitude = uniform_float(rng, cfg.amp_range...),
    ) for i in 1:n_stations]
    texts = [text_fn(rng) for _ in 1:n_stations]
    noise_σ = uniform_float(rng, cfg.noise_range...)

    spec = synthesize_spectrogram(stations, texts, cfg.sr, rng, cfg.spec; noise_σ)
    freqs_out = Float32[s.frequency for s in stations]
    Sample(spec, texts, freqs_out, n_stations)
end

"""
    generate_batch_fast(cfg, batch_size; rng, text_fn, parallel) → Batch

Like generate_batch but uses direct spectrogram synthesis. ~10-30× faster.
"""
function generate_batch_fast(cfg::SamplerConfig, batch_size::Int;
                             rng::AbstractRNG = Random.default_rng(),
                             text_fn::Function = random_text,
                             parallel::Bool = Threads.nthreads() > 1)
    if parallel && Threads.nthreads() > 1
        base_seed = rand(rng, UInt)
        samples = Vector{Sample}(undef, batch_size)
        Threads.@threads for i in 1:batch_size
            rng_i = MersenneTwister(hash(base_seed, UInt(i)))
            @inbounds samples[i] = generate_sample_fast(cfg; rng=rng_i, text_fn)
        end
        collate(samples)
    else
        samples = [generate_sample_fast(cfg; rng, text_fn) for _ in 1:batch_size]
        collate(samples)
    end
end
