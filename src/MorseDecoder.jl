"""
    MorseDecoder

Multi-station Morse code simulation and training data pipeline.

Simulates 1–N CW stations transmitting simultaneously in the 200–800 Hz
band, produces spectrograms as network input, and provides a real-time
streaming interface.  Designed for training an encoder–decoder transformer
that decodes multiple overlapping Morse signals.

# Quick start

```julia
using MorseDecoder, Random

rng = MersenneTwister(42)

# One-shot sample
cfg = SamplerConfig(n_stations_range=1:3)
s   = generate_sample(cfg; rng)           # → Sample (spectrogram + texts)

# Training batch
batch = generate_batch(cfg, 32; rng)      # → Batch (padded tensors)

# Streaming (live band simulation)
stations = [Station(frequency=f, wpm=25) for f in [350, 550, 700]]
stream   = BandStream(stations; rng)
spec, texts = next_chunk!(stream, 16_000) # 2 seconds @ 8 kHz

# Visualization
scene = random_band(rng; n_stations=3)
plot_band(scene, SpectrogramConfig())
```

# Layout
- `morse.jl`       — Morse table, character ↔ index, keying envelope
- `spectrogram.jl` — Short-time FFT → power spectrogram
- `signal.jl`      — Single-station audio synthesis
- `messages.jl`    — Callsign + exchange text generators
- `band.jl`        — Multi-station mixing
- `sampler.jl`     — Training samples, batch collation, BandStream
- `viz.jl`         — CairoMakie plots
"""
module MorseDecoder

using Random
using FFTW: rfft, plan_rfft
using CairoMakie

# ── Source files (order matters) ─────────────────────────────────────────────

include("morse.jl")           # alphabet, keying envelope
include("spectrogram.jl")     # STFT
include("signal.jl")          # Station, synthesize
include("messages.jl")        # callsigns, exchange text
include("band.jl")            # multi-station mixing
include("sampler.jl")         # training data + streaming
include("fast_spectrogram.jl") # direct spectrogram synthesis (no audio/FFT)
include("viz.jl")             # plots
include("model.jl")           # encoder–decoder (Onion + Flux)

# ── Public API ───────────────────────────────────────────────────────────────

export
    # Morse alphabet
    MORSE_TABLE, ALPHABET, NUM_CHARS,
    CHAR_TO_IDX, IDX_TO_CHAR, encode_text, decode_indices,

    # Spectrogram
    SpectrogramConfig, compute_spectrogram, num_bins, num_frames,

    # Station & signal
    Station, synthesize,

    # Messages
    random_callsign, random_message, random_text,

    # Band mixing
    BandScene, mix_stations, random_band,

    # Sampler
    Sample, SamplerConfig, generate_sample,
    Batch, collate, generate_batch,
    synthesize_spectrogram, generate_sample_fast, generate_batch_fast,

    # Streaming
    StationStream, BandStream, next_chunk!, reset!,

    # Visualization
    plot_spectrogram, plot_band, plot_chunk,

    # Model (requires Flux, Onion, Einops)
    SpectrogramEncoder, SpectrogramDecoder, SpectrogramEncoderDecoder,
    prepare_decoder_batch, prepare_training_batch, train_step,
    decode_autoregressive, multi_station_cross_entropy,
    VOCAB_SIZE, START_TOKEN_IDX, PAD_TOKEN_IDX, EOS_TOKEN_IDX

end # module MorseDecoder
