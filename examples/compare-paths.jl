#!/usr/bin/env julia
# Compare AudioPath (ground truth) vs DirectPath spectrograms on the same scene.
# Uses linear band (~10 Hz resolution) and spaced tones so multiple stations show as separate lines.
# Uses a moderate SNR so the audio path looks like real CW (clear lines on noise), and the
# same RNG for both paths so noise is comparable and the comparison is fair.
# Run from MorseDecoder.jl repo root:
#   julia --project=@. examples/compare-paths.jl

using MorseSimulator
using MorseSimulator: DatasetConfig, DirectPath, AudioPath, BandScene,
    generate_transcript, generate_spectrogram, compare_paths,
    SpectrogramResult, STFTConfig, LinearBand, n_bins
using Random
using Statistics
using CairoMakie

outdir = "diagnostic_output"
mkpath(outdir)

sr = 44100
stft = STFTConfig(fft_size=4096, hop_size=128)
lb = LinearBand(fft_size=4096, sample_rate=sr, f_min=200.0, f_max=900.0)
n_b = n_bins(lb)

# Moderate SNR so AudioPath shows clear CW lines (not buried in noise like at -16 dB)
const COMPARE_SNR_RANGE = (0.0, 6.0)  # dB

win_sum = sum(stft.window)
win_sumsq = sum(abs2, stft.window)
tone_gain_db = 10.0 * log10(win_sum^2 / (4.0 * win_sumsq))
println("STFT tone gain: $(round(tone_gain_db; digits=1)) dB (Hann-$(stft.fft_size))")
println("Linear band: $(n_b) bins, ~$(round(sr/stft.fft_size; digits=1)) Hz/bin")
println("Comparison SNR: $(COMPARE_SNR_RANGE) dB (so AudioPath looks like real CW)")
println()

for seed in [42, 123, 456]
    rng = MersenneTwister(seed)
    scene = BandScene(rng; num_stations=3, snr_range=COMPARE_SNR_RANGE, min_tone_separation_Hz=10.0)
    transcript = generate_transcript(rng, scene)

    println("=" ^ 70)
    println("Seed $seed — $(length(scene.stations)) stations (tones 10 Hz apart)")
    for s in scene.stations
        println("  $(s.callsign): tone=$(round(s.tone_freq; digits=1)) Hz, amp=$(round(s.signal_amplitude; digits=4))")
    end
    println("  waveform snr   = $(round(scene.snr_db; digits=1)) dB")
    println("  spec SNR       ≈ $(round(scene.snr_db + tone_gain_db; digits=1)) dB")
    println("  label: $(first(transcript.label, 80))...")

    rng_a = MersenneTwister(seed * 7 + 1)
    rng_d = MersenneTwister(seed * 7 + 2)

    spec_a, _, _ = generate_spectrogram(AudioPath(), rng_a, transcript, scene;
        sample_rate=sr, stft_config=stft, linear_band=lb)
    spec_d, _ = generate_spectrogram(DirectPath(), rng_d, transcript, scene;
        sample_rate=sr, stft_config=stft, linear_band=lb)

    mat_a = spec_a.spectrogram
    mat_d = spec_d.spectrogram

    println("  Audio path:  size=$(size(mat_a)), min=$(round(minimum(mat_a);digits=3)), max=$(round(maximum(mat_a);digits=3)), mean=$(round(mean(mat_a);digits=3))")
    println("  Direct path: size=$(size(mat_d)), min=$(round(minimum(mat_d);digits=3)), max=$(round(maximum(mat_d);digits=3)), mean=$(round(mean(mat_d);digits=3))")

    a_bin_means = vec(mean(mat_a; dims=2))
    d_bin_means = vec(mean(mat_d; dims=2))
    println("  Audio  bin mean range: [$(round(minimum(a_bin_means);digits=3)), $(round(maximum(a_bin_means);digits=3))]")
    println("  Direct bin mean range: [$(round(minimum(d_bin_means);digits=3)), $(round(maximum(d_bin_means);digits=3))]")

    n_frames_common = min(size(mat_a, 2), size(mat_d, 2))
    if n_frames_common > 10
        report = compare_paths(
            SpectrogramResult{Float64}(mat_a[:, 1:n_frames_common], "", sr, 0.0, stft, lb),
            SpectrogramResult{Float64}(mat_d[:, 1:n_frames_common], "", sr, 0.0, stft, lb)
        )
        println("  $report")
    end

    # Plot all frames so every station is visible when they key
    n_show = n_frames_common

    fig = Figure(size=(1600, 800))
    ax1 = Axis(fig[1, 1];
        title="AudioPath — seed=$seed, SNR=$(round(scene.snr_db;digits=1)) dB (3 stations, 10 Hz apart)",
        xlabel="Frame", ylabel="Freq bin")
    heatmap!(ax1, 1:n_show, 1:n_b, mat_a[:, 1:n_show]'; colormap=:viridis)

    ax2 = Axis(fig[2, 1]; title="DirectPath — seed=$seed (same scene)",
        xlabel="Frame", ylabel="Freq bin")
    heatmap!(ax2, 1:n_show, 1:n_b, mat_d[:, 1:n_show]'; colormap=:viridis)

    save(joinpath(outdir, "compare_seed$(seed).png"), fig)
    println("  Saved $outdir/compare_seed$(seed).png")

    fig2 = Figure(size=(800, 400))
    ax3 = Axis(fig2[1, 1];
        title="Mean bin power — seed=$seed, SNR=$(round(scene.snr_db;digits=1)) dB",
        xlabel="Freq bin", ylabel="Mean (log10)")
    barplot!(ax3, collect(1:n_b) .- 0.2, a_bin_means; width=0.4, label="Audio")
    barplot!(ax3, collect(1:n_b) .+ 0.2, d_bin_means; width=0.4, label="Direct")
    axislegend(ax3)
    save(joinpath(outdir, "bins_seed$(seed).png"), fig2)
    println("  Saved $outdir/bins_seed$(seed).png")
    println()
end
