# MorseDecoder.jl

Multi-station Morse (CW) code simulation and training data pipeline. Simulates 1–N stations transmitting simultaneously in the 200–800 Hz band, produces spectrograms as network input, and supports real-time streaming. Designed for training an encoder–decoder transformer that decodes multiple overlapping Morse signals.

## Installation

From the package directory:

```bash
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

Or from Julia:

```julia
using Pkg
Pkg.activate("/path/to/MorseDecoder.jl")
Pkg.instantiate()
```

## Quick start

```julia
using MorseDecoder, Random

rng = MersenneTwister(42)

# One-shot sample (spectrogram + labels)
cfg = SamplerConfig(n_stations_range=1:3)
s   = generate_sample(cfg; rng)

# Training batch
batch = generate_batch(cfg, 32; rng)

# Streaming (live band simulation)
stations = [Station(frequency=f, wpm=25) for f in [350, 550, 700]]
stream   = BandStream(stations; rng)
spec, texts = next_chunk!(stream, 16_000)   # 2 s @ 8 kHz

# Visualization
scene = random_band(rng; n_stations=3)
plot_band(scene, SpectrogramConfig())
```

## Simulating an audio file

You can generate a mixed CW band (multiple stations) and save it as a WAV file, together with a **transcript** of what each station sent.

### 1. Generate a band scene

Either use a **random** scene (stations, frequencies, speeds, and messages are random):

```julia
using MorseDecoder, Random

rng = MersenneTwister(123)
scene = random_band(rng; n_stations=3, sr=8000)
# scene.audio  — Float32 waveform
# scene.sr     — sample rate (Hz)
# scene.texts  — one String per station (what each sent)
# scene.stations — Station parameters (frequency, wpm, etc.)
```

Or define **fixed** stations and messages:

```julia
using MorseDecoder, Random

rng = MersenneTwister(456)
stations = [
    Station(frequency=400f0, wpm=25, amplitude=0.8f0),
    Station(frequency=550f0, wpm=20, amplitude=0.6f0),
]
texts = ["CQ CQ DE SP1ABC SP1ABC K", "SP1ABC DE W1XYZ UR 599 73"]
scene = mix_stations(stations, texts, 8000, rng; noise_σ=0.02f0)
```

### 2. Save audio to WAV and transcript to a file

Add `WAV` if needed, then write the waveform and transcript:

```julia
using WAV

# Save mixed audio
wavwrite(scene.audio, "band.wav"; Fs=scene.sr)

# Save transcript (one line per station: frequency and text)
open("band_transcript.txt", "w") do io
    for (i, (st, txt)) in enumerate(zip(scene.stations, scene.texts))
        println(io, "Station $i @ $(st.frequency) Hz (WPM $(st.wpm)): ", txt)
    end
end
```

### Example transcript

For a random scene you might get something like:

```
Station 1 @ 312.5 Hz (WPM 28): CQ CQ DE DL2XY DL2XY K
Station 2 @ 512.1 Hz (WPM 22): W3AB DE JA1K RST 599 15
Station 3 @ 698.3 Hz (WPM 35): NR 042 137
```

The transcript is the ground truth for training or for checking decoder output.

**Run the example script** (produces **three files**: WAV, transcript TXT, spectrogram PNG):

```bash
julia --project=. examples/simulate_audio.jl
# Custom prefix: julia --project=. examples/simulate_audio.jl myfile
# → myfile.wav, myfile_transcript.txt, myfile.png
```

The PNG shows the waveform and a spectrogram over **200–800 Hz** using the same `SpectrogramConfig` (nfft=512, hop=128) as the model input: compact enough for language models while keeping good time/frequency resolution and avoiding artifacts.

### Longer “audio file” via streaming

To simulate a longer recording (e.g. several minutes) with continuous transmission:

```julia
using MorseDecoder, Random, WAV

rng = MersenneTwister(789)
stations = [Station(frequency=f, wpm=25) for f in [350, 550, 700]]
stream   = BandStream(stations; rng, sr=8000)

# Generate e.g. 60 seconds
sr = 8000
n_seconds = 60
samples_per_chunk = 8 * 16000   # 8 s chunks
all_audio = Float32[]
transcript_lines = String[]

for _ in 1:(n_seconds ÷ 8)
    spec, texts = next_chunk!(stream, samples_per_chunk)
    # Recompute mixed audio for this chunk (stream doesn't expose it directly)
    # So for long files it's easier to use multiple random_band calls and concatenate
end
```

For long files, a simpler approach is to generate several `random_band` scenes and concatenate their `scene.audio` and collect `scene.texts` into a single transcript file (e.g. with time offsets if you track durations).

## Layout

| File            | Role                                      |
|-----------------|-------------------------------------------|
| `morse.jl`      | Morse table, character↔index, keying      |
| `spectrogram.jl`| STFT → power spectrogram                  |
| `signal.jl`     | Single-station audio synthesis            |
| `messages.jl`   | Callsign + exchange text generators       |
| `band.jl`       | Multi-station mixing                      |
| `sampler.jl`    | Training samples, batching, BandStream    |
| `viz.jl`        | CairoMakie plots                          |

## License

See `Project.toml` for authors and version.
