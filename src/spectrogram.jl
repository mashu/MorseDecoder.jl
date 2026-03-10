"""
    spectrogram.jl — Short-time FFT → power spectrogram in a frequency band.

Produces a (freq_bins × time_frames) matrix: one column per hop, only the
bins in [freq_lo, freq_hi] Hz.  We never feed the full FFT (e.g. 257 bins);
for Morse (200–800 Hz) using 100–900 Hz keeps freq_bins small and saves memory.
This is the input representation for the decoder network.
"""

using FFTW: plan_rfft

# ─── Configuration ───────────────────────────────────────────────────────────

struct SpectrogramConfig
    nfft::Int
    hop::Int
    freq_lo::Float32
    freq_hi::Float32
    """Max time frames (optional). When set, training spectrograms are capped to this length to keep GPU memory bounded."""
    max_frames::Union{Int,Nothing}
end

SpectrogramConfig(; nfft=512, hop=128, freq_lo=200f0, freq_hi=800f0, max_frames=nothing) =
    SpectrogramConfig(nfft, hop, freq_lo, freq_hi, max_frames)

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

# ─── Thread-local workspace (reuse window, buffer, FFT plan) ─────────────────

"""Thread-local cache keyed by nfft; parametric on plan type P, no typeof/isa."""
struct SpectrogramWorkspace{P}
    nfft::Int
    win::Vector{Float32}
    buf::Vector{Float32}
    plan::P
end

function SpectrogramWorkspace(nfft::Int)
    buf = Vector{Float32}(undef, nfft)
    plan = plan_rfft(buf)
    SpectrogramWorkspace(nfft, hann(nfft), buf, plan)
end

const spectrogram_workspace_cache = Ref{Vector{Dict{Int, SpectrogramWorkspace}}}([])

"""Get or create thread-local SpectrogramWorkspace for this nfft."""
function thread_local_workspace(nfft::Int)
    tid = Threads.threadid()
    wss = spectrogram_workspace_cache[]
    while length(wss) < tid
        push!(wss, Dict{Int, SpectrogramWorkspace}())
    end
    dict = @inbounds wss[tid]
    get!(dict, nfft) do
        SpectrogramWorkspace(nfft)
    end
end

# ─── Core ────────────────────────────────────────────────────────────────────

"""
    compute_spectrogram(audio, sr, cfg) → Matrix{Float32}

Compute power spectrogram in the [freq_lo, freq_hi] band.
Returns a `(num_bins × num_frames)` matrix — ready to feed to the network.
Uses a thread-local workspace (cached Hann window and FFT plan) for speed when
called from parallel batch generation.
"""
function compute_spectrogram(audio::AbstractVector{<:Real}, sr::Int,
                             cfg::SpectrogramConfig)
    n = length(audio)
    n_frames = num_frames(cfg, n)

    lo_bin = 1 + floor(Int, cfg.freq_lo * cfg.nfft / sr)
    hi_bin = 1 + floor(Int, cfg.freq_hi * cfg.nfft / sr)
    n_bins = hi_bin - lo_bin + 1
    nfft = cfg.nfft
    hop = cfg.hop

    spec = Matrix{Float32}(undef, n_bins, n_frames)
    ws = thread_local_workspace(nfft)
    buf = ws.buf
    win = ws.win
    plan = ws.plan

    @inbounds for f in 1:n_frames
        start = (f - 1) * hop + 1
        stop  = min(start + nfft - 1, n)
        len   = stop - start + 1

        # Windowed frame (zero-pad if at boundary)
        for i in 1:len
            buf[i] = Float32(audio[start + i - 1]) * win[i]
        end
        for i in len+1:nfft
            buf[i] = 0f0
        end

        rfft_result = plan * buf
        for i in lo_bin:hi_bin
            spec[i - lo_bin + 1, f] = abs2(rfft_result[i])
        end
    end

    spec
end
