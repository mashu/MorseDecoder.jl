#!/usr/bin/env julia
# Show sample labels from the same distribution used for training (MorseSimulator).
# Run from repo root:  julia --project=. examples/show_training_examples.jl [N]
# Default N=30. Uses same RNG seed (42) as training.

using MorseDecoder
using MorseSimulator: BandScene, generate_transcript, flat_text
using Random

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
    rng = MersenneTwister(42)

    println("Generating $n training-style transcripts (MorseSimulator, RNG seed 42):\n")
    labels = String[]
    for _ in 1:n
        scene = BandScene(rng; num_stations=rand(rng, 2:4))
        transcript = generate_transcript(rng, scene)
        push!(labels, flat_text(transcript))
    end

    # Prefix counts (first word/token after <START>)
    starts_ts_s1 = count(l -> occursin("<START> [TS] [S1]", l), labels)
    has_cq = count(l -> occursin("CQ", l), labels)
    has_tu = count(l -> occursin("TU", l), labels)
    has_73 = count(l -> occursin("73", l), labels)
    println("Label stats:")
    println("  <START> [TS] [S1]... : $starts_ts_s1 / $n")
    println("  Contains CQ          : $has_cq / $n")
    println("  Contains TU          : $has_tu / $n")
    println("  Contains 73          : $has_73 / $n")
    println()

    println("--- Sample labels (first 200 chars each) ---")
    for (i, label) in enumerate(labels[1:min(10, n)])
        snippet = length(label) > 200 ? label[1:200] * "…" : label
        println("  $i  $snippet")
    end
    if n > 10
        println("  ... and $(n - 10) more")
    end
end

main()
