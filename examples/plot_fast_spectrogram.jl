#!/usr/bin/env julia
# Preview fast_spectrogram training data. Run from repo root:
#   julia --project=. examples/plot_fast_spectrogram.jl
# Saves examples/fast_spectrogram_preview.png

using MorseDecoder
using CairoMakie
using Random

cfg = SamplerConfig(;
    n_stations_range = 1:3,
    spec = SpectrogramConfig(; freq_lo = 100f0, freq_hi = 900f0, max_frames = 512),
)
rng = MersenneTwister(42)
sample = generate_sample_fast(cfg; rng)

# Same layout as plot_chunk: spectrogram + text labels
fig = plot_chunk(
    sample.spectrogram,
    sample.texts,
    cfg.sr,
    cfg.spec;
    title = "Fast spectrogram (100–900 Hz) — $(sample.n_stations) station(s): $(round.(sample.frequencies; digits=0)) Hz",
)
out = joinpath(@__DIR__, "fast_spectrogram_preview.png")
save(out, fig)
@info "Saved" path = out
