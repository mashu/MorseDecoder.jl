# MorseDecoder.jl

[![CI](https://github.com/mashu/MorseDecoder.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/mashu/MorseDecoder.jl/actions/workflows/CI.yml)
[![Codecov](https://codecov.io/gh/mashu/MorseDecoder.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mashu/MorseDecoder.jl)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://mashu.github.io/MorseDecoder.jl/stable)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://mashu.github.io/MorseDecoder.jl/dev)

Training pipeline for a spectrogram encoder–decoder that decodes multi-station CW Morse. Uses **[MorseSimulator.jl](https://github.com/mashu/MorseSimulator.jl)** for data (200–900 Hz mel spectrograms, ~10 Hz resolution, 50 WPM) and **[CTCLoss.jl](https://github.com/mashu/CTCLoss.jl)** for CTC loss and decoding.

**Documentation:** [stable](https://mashu.github.io/MorseDecoder.jl/stable) | [dev](https://mashu.github.io/MorseDecoder.jl/dev)

## Installation

Clone [MorseSimulator.jl](https://github.com/mashu/MorseSimulator.jl) and [CTCLoss.jl](https://github.com/mashu/CTCLoss.jl) next to this repo, then from the MorseDecoder.jl directory:

```bash
julia --project=. -e 'using Pkg; Pkg.add(path="../MorseSimulator.jl"); Pkg.develop(path="../CTCLoss.jl"); Pkg.instantiate()'
```

## Quick start

```julia
using MorseDecoder, Random

rng = MersenneTwister(42)
cfg = SamplerConfig()
s = generate_sample(cfg; rng)
batch = generate_chunked_batch(cfg, 32, rng; max_frames=512)
```

For installation details, spectrogram resolution, label tokens, data throughput, training speed, and troubleshooting, see the **[documentation](https://mashu.github.io/MorseDecoder.jl/stable)**.

## License

See `Project.toml` for authors and version.
