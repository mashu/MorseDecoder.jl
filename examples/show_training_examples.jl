#!/usr/bin/env julia
# Show sample messages from the same distribution used for training.
# Run from repo root:  julia --project=. examples/show_training_examples.jl [N]
# Default N=60. Uses same RNG seed (42) as training so you see the same mix the model sees.
# If "Starts with CQ" is low (e.g. ~10–15%), then CQ-heavy decode is not from data frequency.

using MorseDecoder
using Random

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 60
    rng = MersenneTwister(42)

    println("Generating $n training-style messages (same RNG seed as training, 42):\n")
    messages = [random_message(rng) for _ in 1:n]

    # Count how many start with common prefixes (case-sensitive, strip for "CQ ")
    function startswith_any(s, prefixes)
        for p in prefixes
            startswith(s, p) && return true
        end
        false
    end
    starts_cq = count(m -> startswith_any(first(m), ["CQ ", "CQ"]), messages)
    starts_tu = count(m -> startswith(first(m), "TU"), messages)
    starts_r = count(m -> startswith(first(m), "R "), messages)
    starts_73 = count(m -> startswith(first(m), "73"), messages)
    starts_de = count(m -> startswith(first(m), "DE "), messages)
    starts_nr = count(m -> startswith(first(m), "NR "), messages)
    starts_5 = count(m -> startswith(first(m), "5"), messages)  # 5NN, 599, etc.
    other = n - (starts_cq + starts_tu + starts_r + starts_73 + starts_de + starts_nr + starts_5)
    # (some may overlap, so other can be negative; we just show counts)

    println("First-token / prefix (approx):")
    println("  Starts with CQ...   : $starts_cq / $n  ($(round(100*starts_cq/n; digits=1))%)")
    println("  Starts with TU...   : $starts_tu / $n")
    println("  Starts with R ...   : $starts_r / $n")
    println("  Starts with 73...   : $starts_73 / $n")
    println("  Starts with DE ...  : $starts_de / $n")
    println("  Starts with NR ...  : $starts_nr / $n")
    println("  Starts with 5...    : $starts_5 / $n  (5NN, 599, etc.)")
    println()

    println("Style distribution (from random_message):")
    styles = [last(m) for m in messages]
    for style in unique(styles)
        c = count(==(style), styles)
        println("  $style : $c")
    end
    println()

    println("--- Sample messages (first 40) ---")
    for (i, (text, style)) in enumerate(messages[1:min(40, n)])
        println("  $i  [$style]  $text")
    end
    if n > 40
        println("  ... and $(n - 40) more")
    end
end

main()
