# MorseDecoder.jl

Training pipeline for a spectrogram encoder–decoder that decodes multi-station CW Morse. Uses **[MorseSimulator.jl](https://github.com/mashu/MorseSimulator.jl)** for data: mel spectrograms in the **200–900 Hz** band with ~**10 Hz** frequency resolution (to separate signals) and time resolution suitable for **50 WPM** (dots vs dashes). Labels use the simulator’s special tokens: `<START>`, `<END>`, `[S1]`…`[S6]`, `[TS]`, `[TE]`.

## Installation

**Layout:** Put the MorseSimulator repo next to this one (e.g. `MorseDecoder.jl/` and `MorseSimulator/` in the same parent folder).

From the MorseDecoder.jl directory:

```bash
julia --project=. -e 'using Pkg; Pkg.add("../MorseSimulator"); Pkg.instantiate()'
```

Or from the Julia REPL: `Pkg.activate(".")`, then `Pkg.add("../MorseSimulator")`, then `Pkg.instantiate()`.

## Quick start

```julia
using MorseDecoder, Random

rng = MersenneTwister(42)

# SamplerConfig: 200–900 Hz mel, ~10 Hz resolution, max 512 frames (GPU-friendly)
cfg = SamplerConfig(; max_frames=512)

# One sample (spectrogram + token_ids from simulator label)
s = generate_sample(cfg; rng)

# Training batch
batch = generate_batch_fast(cfg, 32; rng)
# batch.spectrogram  — (n_mels, batch, time)
# batch.targets      — (batch, max_seq) token IDs (<START>, [S1], [TS], [TE], chars, <END>)
```

## Spectrogram and resolution

- **Band:** 200–900 Hz only (Morse CW); no need for wider bandwidth.
- **Frequency resolution:** ~10 Hz (fft_size=4096 @ 44.1 kHz) so signals “up 10” / “down 10” are separable.
- **Time resolution:** hop_size=256 @ 44.1 kHz (~5.8 ms/frame), enough for dots vs dashes up to 50 WPM.

Configured in `SamplerConfig()` via MorseSimulator’s `DatasetConfig` (see `src/sampler.jl`).

### What the direct (analytic) mel path includes

The simulator's **DirectPath** builds the mel spectrogram from Morse events without generating audio. It includes the same high-level variations as the audio path:

| Variation | In DirectPath? | Notes |
|-----------|----------------|--------|
| **Speed (WPM)** | Yes | Per-transmission WPM, jitter, drift → dit/dash timing in events. |
| **Amplitude** | Yes | Per-station `signal_amplitude` (LogNormal) → power ∝ amplitude² in mel. |
| **SNR / noise** | Yes | `ChannelConfig(rng, scene)` from `noise_floor_db`; Gaussian (and optional impulsive) noise power added to mel. |
| **Overlap** | Yes | Transmissions have `time_offset`; events overlap in time; all stations add into the same frames. |
| **QSB fading** | Yes | Applied analytically per-station to frame amplitudes (same propagation params as audio path). |
| **Frequency** | Yes | Each station has `tone_freq`; mel response is computed at that frequency. |

**Caveat:** Neither path currently applies propagation **path loss** to the mixed signal: both use `station.signal_amplitude` as-is (LogNormal variation only). Impulsive noise in the direct path is added as extra constant noise power; the audio path adds actual sample-level impulses.

## Label tokens (MorseSimulator)

Training labels are space-separated: `<START> [TS] [S1] word word [TE] [TS] [S2] ... <END>`. These are mapped to token IDs in `vocab.jl`:

- `<START>`, `<END>` (EOS)
- `[S1]`…`[S6]` — speaker/station
- `[TS]`, `[TE]` — transmission start/end (turn boundaries)
- Characters (A–Z, 0–9, space, `=`, `?`)

## Data throughput and real-time training

The simulator is used in **DirectPath** mode: no audio waveform or FFT, only analytic mel-spectrogram from Morse events. That is the fast path.

- **Per sample:** Each batch item is `BandScene → generate_transcript → encode_transcript → direct mel spectrogram`. So one sample is one scene + one transcript + numeric fill of a (n_mels × n_frames) matrix.
- **Batching:** `generate_batch_fast(cfg, batch_size; parallel=true)` uses `Threads.@threads` to build the batch in parallel (one RNG per thread).
- **Prefetch:** Training uses a channel of pre-generated batches (`--prefetch`, default 256) so the GPU is fed from a queue; you don’t need to generate each batch in real time as long as the producer can refill the channel fast enough.

**Check if you’re data-bound:**

```bash
julia --project=. examples/benchmark_data.jl 64
julia -t 4 --project=. examples/benchmark_data.jl 64
```

Compare “batches/sec” to your training steps/sec (from the “steps_per_sec” log). If batches/sec is lower, increase `--prefetch` or run with more threads (`-t N`). If it’s still too low, options are: reduce `max_frames` or batch size, or add a batch-oriented API in the simulator that reuses buffers (preallocated mel matrices, reuse of scene/transcript structures) and call that from a small “batch producer” functor here. Right now there is no such API; the benchmark tells you whether you need it.

## Troubleshooting

- **Julia 1.11 + path dependency:** With `MorseSimulator = { path = "..." }` in `Project.toml`, `Pkg.instantiate()` can hit `TypeError: in typeassert, expected String, got a value of type Dict{String, Any}` (Julia bug in `get_uuid_name`). Workaround: instantiate once with MorseSimulator removed from `[deps]`, then add it back and run training without calling `instantiate()` again; or use Julia 1.10 and add MorseSimulator via `Pkg.develop(path="...")` from the REPL.
- **CPU-only training:** Run without `--gpu` so CUDA/cuDNN are not loaded; default batch size is 8. Example: `julia --project=. examples/train.jl --steps 1000`.

## Layout

| File             | Role |
|------------------|------|
| `morse.jl`       | Morse table, character↔index |
| `vocab.jl`       | Simulator label ↔ token IDs |
| `spectrogram.jl` | STFT (e.g. for inference on real WAV) |
| `model.jl`       | Encoder–decoder (Onion + Flux) |
| `sampler.jl`     | MorseSimulator-based Sample/Batch |
| `viz.jl`         | CairoMakie plots |

## License

See `Project.toml` for authors and version.
