#!/usr/bin/env julia
# Report distribution of [TS]…[TE] transmission lengths (frames) from the simulator.
# Run from repo root: julia --project=. examples/transmission_length_distribution.jl [n_samples]
#
# Output: min, max, mean, percentiles, counts above 512/1024, and a simple histogram.

using MorseDecoder
using MorseSimulator: DatasetConfig, DirectPath
using Random
using Statistics

function segment_lengths(sample)
    ids = sample.token_ids
    timing = sample.token_timing
    segs = transmission_segments(ids)
    T_total = size(sample.spectrogram, 2)
    lengths = Int[]
    for (tok_start, tok_end) in segs
        f_start = clamp(timing.token_start_frames[tok_start], 1, T_total)
        f_end = clamp(timing.token_end_frames[tok_end], 1, T_total)
        n = f_end - f_start + 1
        n > 0 && push!(lengths, n)
    end
    lengths
end

function main()
    n_samples = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500
    cfg = DatasetConfig(;
        path = DirectPath(),
        sample_rate = 44100,
        fft_size = 4096,
        hop_size = 128,
        f_min = 200.0,
        f_max = 900.0,
        stations = 2:4,
    )
    rng = MersenneTwister(42)

    all_lengths = Int[]
    n_no_timing = 0
    n_empty = 0

    for _ in 1:n_samples
        sample = generate_sample(cfg; rng)
        # TokenTiming from simulator has token_start_frames, token_end_frames
        if !hasproperty(sample.token_timing, :token_start_frames)
            n_no_timing += 1
            continue
        end
        lens = segment_lengths(sample)
        if isempty(lens)
            n_empty += 1
            continue
        end
        append!(all_lengths, lens)
    end

    println("Transmission [TS]…[TE] length distribution (frames)")
    println("=" ^ 60)
    println("Samples: $n_samples  (skipped no timing: $n_no_timing, empty segments: $n_empty)")
    println("Total segments: $(length(all_lengths))")

    if isempty(all_lengths)
        println("No segments with timing; cannot compute distribution.")
        return
    end

    sort!(all_lengths)
    n = length(all_lengths)
    println()
    println("Frames per transmission:")
    println("  min:     $(minimum(all_lengths))")
    println("  max:     $(maximum(all_lengths))")
    println("  mean:    $(round(mean(all_lengths); digits=1))")
    println("  median:  $(all_lengths[div(n, 2) + 1])")
    p50 = all_lengths[max(1, div(n, 2))]
    p90 = all_lengths[max(1, div(9n, 10))]
    p95 = all_lengths[max(1, div(19n, 20))]
    p99 = all_lengths[max(1, div(99n, 100))]
    println("  p90:     $p90")
    println("  p95:     $p95")
    println("  p99:     $p99")
    println()
    above_512 = count(>(512), all_lengths)
    above_1024 = count(>(1024), all_lengths)
    println("Segments longer than max_frames:")
    println("  > 512:   $above_512 ($(round(100 * above_512 / n; digits=1))%)  → would be sub-chunked with max_frames=512")
    println("  > 1024:  $above_1024 ($(round(100 * above_1024 / n; digits=1))%)  → would be sub-chunked with max_frames=1024")
    println()

    # Simple histogram (log-spaced bins)
    bins = [0, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 100000]
    hist = zeros(Int, length(bins) - 1)
    for L in all_lengths
        for i in 1:length(bins)-1
            if L >= bins[i] && L < bins[i+1]
                hist[i] += 1
                break
            end
        end
    end
    max_count = maximum(hist)
    bar_width = 50
    println("Histogram (frames):")
    for i in 1:length(bins)-1
        lo, hi = bins[i], bins[i+1]
        w = max_count > 0 ? max(0, round(Int, bar_width * hist[i] / max_count)) : 0
        bar = "█" ^ w * "░" ^ (bar_width - w)
        println("  [$(lpad(lo, 4)) - $(lpad(hi, 5))): $(lpad(hist[i], 5))  $bar")
    end
end

main()
