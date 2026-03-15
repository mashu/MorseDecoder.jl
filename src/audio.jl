"""
    audio.jl — Live audio input (mic) for inference.

Uses PortAudio for capture and yields spectrogram chunks compatible with
`decode_conversation`. No dependency on the simulator: you can switch from
simulator chunks to mic (or file) by passing a different chunk iterable.

**Dependencies:** PortAudio, SampledSignals (for stream read API).
**Match training:** Use the same sample rate as training (e.g. 44100) and a
spectrogram config that yields the same n_bins (e.g. 40). See `mic_spectrogram_config`.
"""

using PortAudio
using SampledSignals: read, samplerate

# ─── Spectrogram config that matches typical model (40 bins, 200–620 Hz at 44.1 kHz) ───
# Training uses 40 mel bins in 200–900 Hz. Linear FFT with nfft=4096 at 44100 Hz has
# ~10.77 Hz/bin; 40 bins over 200–620 Hz gives 40 bins. Same hop as training (128).
const DEFAULT_MIC_HOP = 128
const DEFAULT_MIC_NFFT = 4096

"""
    mic_spectrogram_config(sample_rate; n_bins, hop, nfft)

SpectrogramConfig for real-time mic input that yields `n_bins` (default 40) in the
CW band, to match a model trained with 40 mel bins. At 44100 Hz with nfft=4096,
freq_hi is set so that the linear band has 40 bins; use the same sample_rate as training.
"""
function mic_spectrogram_config(sample_rate::Int; n_bins::Int = 40, hop::Int = DEFAULT_MIC_HOP, nfft::Int = DEFAULT_MIC_NFFT)
    freq_lo = 200f0
    bin_width_hz = Float32(sample_rate / nfft)
    freq_hi = freq_lo + (n_bins - 1) * bin_width_hz
    SpectrogramConfig(; nfft, hop, freq_lo, freq_hi)
end

# ─── Audio buffer → mono Float32 (dispatch, no isa check) ────────────────────

"""Convert multi-channel read buffer to mono Float32 vector."""
audio_to_mono_float(buf::AbstractMatrix) = Float32.(vec(buf[1, :]))

"""Convert single-channel read buffer to mono Float32 vector."""
audio_to_mono_float(buf::AbstractVector) = Float32.(vec(buf))

# ─── Mic stream → iterable of spectrogram chunks ─────────────────────────────

"""
    MicSpectrogramSource(stream, spec_config; chunk_seconds)

Wraps an open PortAudio input stream and yields spectrogram chunks (one per iteration)
suitable for `decode_conversation`. Each chunk is (n_bins × n_frames) Float32 in
model scale (peak-norm + log10). Close the stream when done (e.g. with `close(stream)`).

**Usage:**
    stream = PortAudioStream(1, 0; samplerate = 44100)
    cfg = mic_spectrogram_config(44100)
    src = MicSpectrogramSource(stream, cfg; chunk_seconds = 1.0)
    tokens = decode_conversation(model, src, gpu)
    close(stream)
"""
mutable struct MicSpectrogramSource{S,C}
    stream::S
    spec_config::C
    chunk_seconds::Float64
end

function MicSpectrogramSource(stream, spec_config::SpectrogramConfig; chunk_seconds::Float64 = 1.0)
    MicSpectrogramSource(stream, spec_config, chunk_seconds)
end

# Single implementation; both iterate signatures delegate here.
function read_mic_chunk(src::MicSpectrogramSource)
    isopen(src.stream) || return nothing
    sr = Int(samplerate(src.stream))
    n_samples = max(1, round(Int, src.chunk_seconds * sr))
    buf = read(src.stream, n_samples)
    audio = audio_to_mono_float(buf)
    length(audio) < src.spec_config.nfft && return nothing
    spec = compute_spectrogram(audio, sr, src.spec_config)
    spec_scaled = spectrogram_to_model_scale(spec)
    (spec_scaled, nothing)
end

Base.iterate(src::MicSpectrogramSource) = read_mic_chunk(src)
Base.iterate(src::MicSpectrogramSource, _) = read_mic_chunk(src)

Base.IteratorSize(::Type{<:MicSpectrogramSource}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:MicSpectrogramSource}) = Base.EltypeUnknown()

"""
    open_mic_input(; sample_rate, input_channels)

Open the default microphone as an input-only PortAudioStream. Use with
`MicSpectrogramSource(stream, mic_spectrogram_config(sample_rate); chunk_seconds)`.
Close the stream when done.
"""
function open_mic_input(; sample_rate::Int = 44100, input_channels::Int = 1)
    PortAudioStream(input_channels, 0; samplerate = Float64(sample_rate), warn_xruns = false)
end
