"""
Visualization utilities using CairoMakie.

Spectrogram and waveform plots for data exploration (no model dependency).
"""

"""
    plot_waveform(signal::MorseSignal) → Figure
"""
function plot_waveform(signal::MorseSignal)
    audio = Array(signal.audio)
    sr    = signal.config.sample_rate
    t     = range(0, step = 1 / sr, length = length(audio))

    fig = Figure(size = (1000, 300))
    ax  = Axis(fig[1, 1];
        title  = "Waveform — \"$(signal.text)\"",
        xlabel = "Time (s)",
        ylabel = "Amplitude")
    lines!(ax, Float64.(t), Float64.(audio); color = :steelblue, linewidth = 0.5)
    fig
end
