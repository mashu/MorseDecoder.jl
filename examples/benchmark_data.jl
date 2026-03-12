#!/usr/bin/env julia
# Benchmark data generation: samples/sec and batches/sec (single- vs multi-thread).
# Run from repo root:
#   julia --project=. examples/benchmark_data.jl [batch_size]
#   julia -t 4 --project=. examples/benchmark_data.jl 64
#
# Use -t N (N>1) to test parallel batch generation. If "batches/sec" is below
# your training steps/sec, the GPU will be data-starved unless prefetch is large enough.

using MorseDecoder
using Random

function main()
    batch_size = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 64
    cfg = SamplerConfig(; max_frames=512)
    rng = MersenneTwister(123)
    n_warmup = 2
    n_batches = 10

    # Warmup
    for _ in 1:n_warmup
        generate_batch_fast(cfg, batch_size; rng)
    end

    # Single-thread
    rng1 = MersenneTwister(456)
    t0 = time()
    for _ in 1:n_batches
        generate_batch_fast(cfg, batch_size; rng=rng1, parallel=false)
    end
    elapsed_st = time() - t0
    batches_per_sec_st = n_batches / elapsed_st
    samples_per_sec_st = (n_batches * batch_size) / elapsed_st

    # Multi-thread (if available)
    n_threads = Threads.nthreads()
    if n_threads > 1
        rng2 = MersenneTwister(789)
        t0 = time()
        for _ in 1:n_batches
            generate_batch_fast(cfg, batch_size; rng=rng2, parallel=true)
        end
        elapsed_mt = time() - t0
        batches_per_sec_mt = n_batches / elapsed_mt
        samples_per_sec_mt = (n_batches * batch_size) / elapsed_mt
    end

    println("Data generation benchmark (batch_size=$batch_size, max_frames=512)")
    println("  Threads: ", n_threads)
    println("  Single-thread: ", round(samples_per_sec_st; digits=1), " samples/sec  ", round(batches_per_sec_st; digits=2), " batches/sec")
    if n_threads > 1
        println("  Multi-thread:  ", round(samples_per_sec_mt; digits=1), " samples/sec  ", round(batches_per_sec_mt; digits=2), " batches/sec")
    end
    println()
    println("If training does more steps/sec than batches/sec above, increase --prefetch or use more threads (-t N).")
end

main()
