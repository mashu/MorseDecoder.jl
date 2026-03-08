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

    fig = Figure(size = (900, 350))
    ax  = Axis(fig[1, 1]; title, xlabel = "Time (s)", ylabel = "Frequency (Hz)")
    heatmap!(ax, collect(times), collect(freqs),
             log10.(Float32.(spec) .+ 1f-10); colormap)
    fig
end

# ─── Band overview ───────────────────────────────────────────────────────────

"""
    plot_band(scene, cfg; title) → Figure

Plot the mixed audio waveform and spectrogram of a `BandScene`, with station
frequencies marked as horizontal lines.
"""
function plot_band(scene::BandScene, cfg::SpectrogramConfig;
                   title::String = "Band — $(scene.sr) Hz")
    spec  = compute_spectrogram(scene.audio, scene.sr, cfg)
    n_bins, n_frames = size(spec)

    times = range(0, step = cfg.hop / scene.sr, length = n_frames)
    freqs = range(cfg.freq_lo, cfg.freq_hi, length = n_bins)
    t_audio = range(0, step = 1 / scene.sr, length = length(scene.audio))

    fig = Figure(size = (1000, 600))

    # Waveform
    ax1 = Axis(fig[1, 1]; title = "$title — waveform",
               xlabel = "Time (s)", ylabel = "Amplitude")
    lines!(ax1, collect(t_audio), scene.audio; color = :steelblue, linewidth = 0.3)

    # Spectrogram + station markers
    ax2 = Axis(fig[2, 1]; title = "$title — spectrogram",
               xlabel = "Time (s)", ylabel = "Frequency (Hz)")
    heatmap!(ax2, collect(times), collect(freqs),
             log10.(Float32.(spec) .+ 1f-10); colormap = :inferno)

    colors = [:cyan, :lime, :magenta, :yellow, :orange, :red]
    for (i, s) in enumerate(scene.stations)
        c = colors[mod1(i, length(colors))]
        hlines!(ax2, [s.frequency]; color = c, linewidth = 1.5, linestyle = :dash,
                label = "$(round(s.frequency; digits=0)) Hz: \"$(first(scene.texts[i], 20))…\"")
    end
    axislegend(ax2; position = :rt, labelsize = 9, framevisible = false)

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

    fig = Figure(size = (900, 350))
    ax  = Axis(fig[1, 1]; title, xlabel = "Time (s)", ylabel = "Frequency (Hz)")
    heatmap!(ax, collect(times), collect(freqs),
             log10.(Float32.(spec) .+ 1f-10); colormap = :inferno)

    # Annotate texts in the margin
    label = join(["[$i] $(first(t, 30))" for (i, t) in enumerate(texts)], "  |  ")
    Label(fig[0, 1], label; fontsize = 10, halign = :left)

    fig
end
