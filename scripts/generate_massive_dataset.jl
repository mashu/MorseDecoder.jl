#!/usr/bin/env julia
"""
Generate a massive Morse training dataset covering the full distribution of:
  • WPM (slow to very fast)
  • Noise levels (clean to heavy)
  • Messages (all contest types, many exchanges)
  • Tones (pitch range)
  • Per-word drift (realistic speed variation)

Usage:
    julia --project=. scripts/generate_massive_dataset.jl [options]

Options:
    --n N              Total samples (default: 100000)
    --output DIR       Output directory (default: dataset_massive)
    --sample-rate R    Sample rate in Hz (default: 2000; use 44100 for realistic then downsample on load)
    --seed S           Random seed (default: 42)
    --stratify         Balance samples across contest types (round-robin)
    --quick            Small run: 5000 samples, narrow ranges (for testing)

Storage: 2000 Hz → ~24 KB per ~6s WAV, 100k ≈ 2.4 GB. 44100 Hz → ~22× larger (100k ≈ 53 GB).
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MorseDecoder
using Random

function parse_args()
    n_samples   = 100_000
    output_dir  = "dataset_massive"
    seed        = 42
    stratify    = false
    quick       = false
    sample_rate = 2000

    i = 1
    while i ≤ length(ARGS)
        if ARGS[i] == "--n"
            n_samples = parse(Int, ARGS[i + 1]); i += 2
        elseif ARGS[i] == "--output"
            output_dir = ARGS[i + 1]; i += 2
        elseif ARGS[i] == "--seed"
            seed = parse(Int, ARGS[i + 1]); i += 2
        elseif ARGS[i] == "--stratify"
            stratify = true; i += 1
        elseif ARGS[i] == "--quick"
            quick = true; i += 1
        elseif ARGS[i] == "--sample-rate"
            sample_rate = parse(Int, ARGS[i + 1]); i += 2
        else
            i += 1
        end
    end

    (; n_samples, output_dir, seed, stratify, quick, sample_rate)
end

function main()
    (; n_samples, output_dir, seed, stratify, quick, sample_rate) = parse_args()

    rng = MersenneTwister(seed)

    # Full distribution: wide WPM, pitch, noise, amplitude, drift
    if quick
        n_samples = min(n_samples, 5_000)
        pitch_range = 400:800
        wpm_range   = 20:40
        noise_range = 0:150
        amp_range   = 40:120
        max_drift   = 4f0
    else
        pitch_range = 200:1000   # tones from low to high
        wpm_range   = 10:55      # slow ragchew to very fast contest
        noise_range = 0:350      # clean to heavy QRM
        amp_range   = 15:150     # weak to strong signal
        max_drift   = 8f0        # larger per-word speed variation
    end

    println("╔══════════════════════════════════════════════════════════════════╗")
    println("║      MorseDecoder.jl — Massive Training Dataset                 ║")
    println("╠══════════════════════════════════════════════════════════════════╣")
    println("║  Samples     : $n_samples")
    println("║  Output      : $output_dir/")
    println("║  Sample rate : $sample_rate Hz (Int16 WAV)")
    println("║  Pitch       : $pitch_range Hz")
    println("║  WPM         : $wpm_range (max drift ±$(Int(max_drift)) WPM)")
    println("║  Noise       : $noise_range")
    println("║  Amplitude   : $amp_range %")
    println("║  Stratify    : $stratify")
    println("║  Seed        : $seed")
    println("╚══════════════════════════════════════════════════════════════════╝")
    if sample_rate >= 44100
        println("  → 44.1 kHz WAVs are ~22× larger than 2 kHz; 100k ≈ 53 GB. Load with load_dataset(\"$output_dir\") and audio is resampled to 2000 Hz for training.")
    end
    println()

    config = DatasetConfig(;
        n_samples,
        output_dir,
        sample_rate,
        pitch_range,
        wpm_range,
        noise_range,
        amp_range,
        max_drift,
        contest     = nothing,
        stratify_contests = stratify,
    )

    @time entries = generate_dataset(config; rng)

    stats = dataset_stats(entries)
    println("\n── Dataset Statistics ───────────────────────────────────────────")
    println("  Total samples   : $(stats.n_samples)")
    println("  Total audio     : $(round(stats.total_audio_h; digits=2)) hours")
    println("  Total WAV size  : $(round(stats.total_wav_mb; digits=1)) MB")
    println("  Mean duration   : $(round(stats.duration_mean; digits=2))s " *
            "(σ = $(round(stats.duration_std; digits=2))s)")
    println("  WPM range       : $(stats.wpm_range)")
    println("  Pitch range     : $(stats.pitch_range)")
    println("  Noise range     : $(stats.noise_range)")
    println("  Contest distribution:")
    for (k, v) in sort(collect(stats.contest_dist); by = last, rev = true)
        pct = round(100v / stats.n_samples; digits = 1)
        println("    $(rpad(k, 18)) : $v ($pct%)")
    end

    println("\n✓ Done! Massive dataset ready at $(output_dir)/")
    println("  Load with: loader = load_dataset(\"$(output_dir)\")")
end

main()
