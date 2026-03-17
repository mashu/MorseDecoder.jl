"""
    viz.jl — Spectrogram and band visualization with CairoMakie.
"""

# ─── Encoder input (what the network sees) ───────────────────────────────────

"""
    plot_encoder_input(spec, sr, cfg; title, colormap) → Figure

Plot the **exact** spectrogram the encoder receives: (n_bins × n_frames) in **model scale**
(peak-normalized, log10). Use this to see the visual representation of the network input.

- `spec`: (bins × frames) already in model scale (e.g. `sample.spectrogram` from the simulator,
  or `spectrogram_to_model_scale(compute_spectrogram(...))` for real audio). Values are typically ≈ -8 to 0.
- No extra log is applied; the color scale is the actual input value.
"""
function plot_encoder_input(spec::AbstractMatrix{<:Real}, sr::Int,
                            cfg::SpectrogramConfig;
                            title::String = "Encoder input (model scale)",
                            colormap = :viridis)
    n_bins, n_frames = size(spec)
    times = range(0, step = cfg.hop / sr, length = n_frames)
    freqs = range(cfg.freq_lo, cfg.freq_hi, length = n_bins)
    # spec is already log10; plot as-is (z[i,j] = spec value at (freq_i, time_j))
    z = Float32.(spec)'

    fig = Figure(size = (950, 350))
    ax  = Axis(fig[1, 1]; title, xlabel = "Time (s)", ylabel = "Frequency (Hz)")
    hm = heatmap!(ax, collect(times), collect(freqs), z; colormap)
    ylims!(ax, cfg.freq_lo, cfg.freq_hi)
    Colorbar(fig[1, 2], hm; label = "log10 (model input)")
    fig
end

# ─── Spectrogram heatmap (linear power) ─────────────────────────────────────

"""
    plot_spectrogram(spec, sr, cfg; title, colormap) → Figure

Plot a spectrogram matrix as a heatmap. Expects **linear** power (e.g. raw FFT power);
applies log10 for display. Frequency axis spans [freq_lo, freq_hi], time from hop/sr.
For data already in model scale (simulator or after spectrogram_to_model_scale), use
`plot_encoder_input` instead.
"""
function plot_spectrogram(spec::AbstractMatrix{<:Real}, sr::Int,
                          cfg::SpectrogramConfig;
                          title::String = "Spectrogram",
                          colormap = :inferno)
    n_bins, n_frames = size(spec)
    times = range(0, step = cfg.hop / sr, length = n_frames)
    freqs = range(cfg.freq_lo, cfg.freq_hi, length = n_bins)
    # Makie heatmap(x,y,z): z[i,j] at (x[i], y[j]); we want time on x, freq on y → z (n_frames × n_bins)
    z = (log10.(Float32.(spec) .+ 1f-10))'

    fig = Figure(size = (900, 350))
    ax  = Axis(fig[1, 1]; title, xlabel = "Time (s)", ylabel = "Frequency (Hz)")
    heatmap!(ax, collect(times), collect(freqs), z; colormap)
    ylims!(ax, cfg.freq_lo, cfg.freq_hi)
    fig
end

# ─── Stream chunk plot ───────────────────────────────────────────────────────

"""
    plot_chunk(spec, texts, sr, cfg; title) → Figure

Quick visualization of one streaming chunk: spectrogram + text annotations.
"""
function plot_chunk(spec::AbstractMatrix{<:Real}, texts::AbstractVector{<:AbstractString},
                    sr::Int, cfg::SpectrogramConfig;
                    title::String = "Stream chunk")
    n_bins, n_frames = size(spec)
    times = range(0, step = cfg.hop / sr, length = n_frames)
    freqs = range(cfg.freq_lo, cfg.freq_hi, length = n_bins)
    z = (log10.(Float32.(spec) .+ 1f-10))'

    fig = Figure(size = (900, 350))
    ax  = Axis(fig[1, 1]; title, xlabel = "Time (s)", ylabel = "Frequency (Hz)")
    heatmap!(ax, collect(times), collect(freqs), z; colormap = :inferno)
    ylims!(ax, cfg.freq_lo, cfg.freq_hi)

    # Annotate texts in the margin
    label = join(["[$i] $(first(t, 30))" for (i, t) in enumerate(texts)], "  |  ")
    Label(fig[0, 1], label; fontsize = 10, halign = :left)

    fig
end
