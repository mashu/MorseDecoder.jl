# Quick start

```julia
using MorseDecoder, Random

rng = MersenneTwister(42)
cfg = SamplerConfig()   # 200–900 Hz, ~10 Hz resolution
s   = generate_sample(cfg; rng)   # one full conversation
batch = generate_chunked_batch(cfg, 32, rng; max_frames=512)   # one batch of chunks
```

- **ChunkedConversation(spec, token_ids, max_frames)** — iterable over `Sample` chunks of one conversation (split at [TS]/[TE]).
- **ChunkedSampleSource(cfg, max_frames; rng)** — infinite iterator of chunked samples.
- **ChunkedBatchSource(cfg, batch_size, max_frames; rng)** — infinite iterator of `Batch` for training.
- **decode_conversation(model, chunks, to_device; ...)** — decode a full conversation from an iterable of spectrogram chunks (chunk-by-chunk with continuation; supports streaming).

## Spectrogram and resolution

- **Band:** 200–900 Hz only (Morse CW).
- **Frequency resolution:** ~10 Hz (fft_size=4096 @ 44.1 kHz) so signals “up 10” / “down 10” are separable.
- **Time resolution:** hop_size=128 @ 44.1 kHz (~2.9 ms/frame), 4–5+ frames per dot for dots vs dashes up to ~80 WPM.

Configured in `SamplerConfig()` via MorseSimulator's `DatasetConfig` (see `src/sampler.jl`).

## Label tokens (MorseSimulator)

Training labels are space-separated: `<START> [TS] [S1] word word [TE] [TS] [S2] ... <END>`. Mapped to token IDs in the package:

- `<START>`, `<END>` (EOS)
- `[S1]`…`[S6]` — speaker/station
- `[TS]`, `[TE]` — transmission start/end (turn boundaries)
- Characters (A–Z, 0–9, space, `=`, `?`)

## Data throughput and training

The simulator is used in **DirectPath** mode: analytic mel-spectrogram from Morse events (no audio waveform). Batching uses `Threads.@threads`; training can use a prefetch channel of batches so the GPU is fed from a queue.

**Check if you're data-bound:**

```bash
julia --project=. examples/benchmark_data.jl 64
julia -t 4 --project=. examples/benchmark_data.jl 64
```

Compare “batches/sec” to your training steps/sec. If batches/sec is lower, increase `--prefetch` or run with more threads (`-t N`).

## Training speed

Run a short benchmark with timing breakdown:

```bash
julia -t 4 --project=. examples/train.jl --gpu --benchmark 50
```

Logs show **Timing (ms)** for: **data**, **transfer**, **forward_backward**, **accum_update**, and a **Bottleneck** hint.

| If the bottleneck is… | Try |
|------------------------|-----|
| **Data** | More threads (`julia -t 4`), ensure `--prefetch` ≥ 2. |
| **Forward/backward (GPU)** | Increase `--batch` if GPU memory allows; if OOM, use `--batch 32` or `--max-frames 256`. |
| **Decode / save** | Reduce `--decode-every` or `--save-every`. |
| **Too many frames** | Use `--max-frames 256` (or 384). |

Always run with **multiple threads** when using prefetch: `julia -t 4 --project=. examples/train.jl --gpu --steps 5000`.

## CPU-only training

Run without `--gpu`; default batch size is 8. Example: `julia --project=. examples/train.jl --steps 1000`.
