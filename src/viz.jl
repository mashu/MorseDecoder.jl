"""
    viz.jl — Spectrogram and band visualization with CairoMakie.
"""

# ─── Spectrogram heatmap ────────────────────────────────────────────────────

"""
    plot_spectrogram(spec, sr, cfg; title, colormap) → Figure

Plot a spectrogram matrix as a heatmap.  Frequency axis spans [freq_lo, freq_hi],
time axis is derived from hop size.
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
