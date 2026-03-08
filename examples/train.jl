#!/usr/bin/env julia
# Train the spectrogram encoder–decoder. Run from repo root:
#   julia --project=. examples/train.jl [--gpu] [--steps 10000] [--batch 32] [--accum 1]
#   [--checkpoint-dir checkpoints] [--save-every 1000] [--prefetch 4] [--lr 1e-4]
#   [--decode-every 100]  decode on fixed validation spectrogram every N steps (0 = only at end)
#   [--dim 128] [--n-layers 5] [--n-heads 4]  smaller dim ok; n_layers >= 5 for RoPE
#
# Uses MorseDecoder sampler for batches. For GPU: pass --gpu; use --batch to fill the GPU
# and --accum for gradient accumulation (effective batch = batch * accum).
# Checkpoints (model + optimiser state, on CPU) are saved every --save-every steps; if
# checkpoint_latest.jld2 exists in --checkpoint-dir, training resumes from it on GPU.
# --prefetch N: producer task pre-generates up to N batches so GPU is not waiting on data.

using MorseDecoder
using Flux
using Random
using CUDA
using cuDNN
using JLD2

function parse_args()
    gpu = "--gpu" in ARGS
    steps = 10_000
    batch_size = gpu ? 32 : 8
    accum_steps = 1
    checkpoint_dir = "checkpoints"
    save_every = 1000
    prefetch = 4
    lr = 1f-4
    decode_every = 0
    dim = 128
    n_layers = 5   # at least 5 for RoPE
    n_heads = 4
    for i in eachindex(ARGS)
        if ARGS[i] == "--steps" && i < length(ARGS)
            steps = parse(Int, ARGS[i + 1])
        elseif ARGS[i] == "--batch" && i < length(ARGS)
            batch_size = parse(Int, ARGS[i + 1])
        elseif ARGS[i] == "--accum" && i < length(ARGS)
            accum_steps = parse(Int, ARGS[i + 1])
        elseif ARGS[i] == "--checkpoint-dir" && i < length(ARGS)
            checkpoint_dir = ARGS[i + 1]
        elseif ARGS[i] == "--save-every" && i < length(ARGS)
            save_every = parse(Int, ARGS[i + 1])
        elseif ARGS[i] == "--prefetch" && i < length(ARGS)
            prefetch = parse(Int, ARGS[i + 1])
        elseif ARGS[i] == "--lr" && i < length(ARGS)
            lr = parse(Float32, ARGS[i + 1])
        elseif ARGS[i] == "--decode-every" && i < length(ARGS)
            decode_every = parse(Int, ARGS[i + 1])
        elseif ARGS[i] == "--dim" && i < length(ARGS)
            dim = parse(Int, ARGS[i + 1])
        elseif ARGS[i] == "--n-layers" && i < length(ARGS)
            n_layers = max(5, parse(Int, ARGS[i + 1]))  # minimum 5 for RoPE
        elseif ARGS[i] == "--n-heads" && i < length(ARGS)
            n_heads = parse(Int, ARGS[i + 1])
        end
    end
    (; gpu, steps, batch_size, accum_steps, checkpoint_dir, save_every, prefetch, lr, decode_every, dim, n_layers, n_heads)
end

function build_model(n_bins::Int, chunk_frames::Int; dim=128, n_heads=4, n_layers=2, max_stations::Int=5)
    token_dim = n_bins * chunk_frames
    encoder = SpectrogramEncoder(token_dim, dim, n_heads, n_layers, chunk_frames)
    decoder = SpectrogramDecoder(VOCAB_SIZE, dim, n_heads, n_layers; max_stations)
    SpectrogramEncoderDecoder(encoder, decoder)
end

const CHUNK_FRAMES = 4

"""Run decode on one spectrogram, n_stations outputs; print raw first 8 tokens and decoded string."""
function run_decode_test(model, batch, device; n_stations::Int=2, max_len::Int=32)
    test_spec = batch.spectrogram[:, 1:1, :]
    test_spec = device(test_spec)
    ids = decode_autoregressive(model, test_spec, n_stations; max_len, to_device=device)
    ids_cpu = cpu(ids)
    for k in 1:n_stations
        seq = [ids_cpu[i, k] for i in 1:size(ids_cpu, 1)]
        println("    station $k (raw first 8): ", Int.(seq[1:min(8, end)]))
        s = token_ids_to_string(seq)
        println("    station $k: ", isempty(s) ? "(empty)" : s)
    end
end

"""Turn decoded token id sequence into a string: chars 1:NUM_CHARS, BLANK→space, stop at PAD/START."""
function token_ids_to_string(ids::AbstractVector{<:Integer})
    buf = Char[]
    for i in ids
        i == PAD_TOKEN_IDX && break
        i == START_TOKEN_IDX && continue
        if 1 <= i <= NUM_CHARS
            push!(buf, IDX_TO_CHAR[i])
        elseif i == BLANK_TOKEN
            push!(buf, ' ')
        end
    end
    String(buf)
end

function save_checkpoint(path::String, step::Int, n_bins::Int, model, opt)
    mkpath(dirname(path))
    model_cpu = cpu(model)
    opt_cpu = cpu(opt)
    model_state = Flux.state(model_cpu)
    jldsave(path; step, n_bins, chunk_frames=CHUNK_FRAMES, model_state, optimiser_state=opt_cpu)
    println("  checkpoint saved at step $step -> $path")
end

function load_checkpoint(path::String)
    isfile(path) || return nothing
    jldopen(path, "r") do f
        (; step = f["step"], n_bins = f["n_bins"], chunk_frames = f["chunk_frames"],
          model_state = f["model_state"], optimiser_state = f["optimiser_state"])
    end
end

function main()
    args = parse_args()
    rng = MersenneTwister(42)
    device = args.gpu ? gpu : cpu
    checkpoint_path = joinpath(args.checkpoint_dir, "checkpoint_latest.jld2")

    # Data: 1–3 stations, same spectrogram config as in SamplerConfig
    cfg = SamplerConfig(; n_stations_range=1:3, spec=SpectrogramConfig())
    batch = generate_batch(cfg, args.batch_size; rng)
    n_bins = size(batch.spectrogram, 1)

    loaded = load_checkpoint(checkpoint_path)
    step_start = 1
    if loaded !== nothing
        (; step, n_bins, chunk_frames, model_state, optimiser_state) = loaded
        step_start = step + 1
        model = build_model(n_bins, chunk_frames; dim=args.dim, n_heads=args.n_heads, n_layers=args.n_layers)
        Flux.loadmodel!(model, model_state)
        opt = optimiser_state
        if args.gpu
            model = device(model)
            opt = device(opt)
        end
        println("Resumed from $checkpoint_path (step $step -> continuing from $step_start)")
    else
        model = build_model(n_bins, CHUNK_FRAMES; dim=args.dim, n_heads=args.n_heads, n_layers=args.n_layers)
        if args.gpu
            model = device(model)
        end
        opt = Flux.setup(Adam(args.lr), model)
    end

    effective_batch = args.batch_size * args.accum_steps
    # Fixed validation batch for decode (same inputs every time so we see real progress)
    rng_val = MersenneTwister(123)
    val_batch = generate_batch(cfg, 1; rng=rng_val)
    n_val_stations = min(2, maximum(val_batch.n_stations))

    println("Training for $(args.steps) steps on $(args.gpu ? "GPU" : "CPU") (batch=$(args.batch_size), accum=$(args.accum_steps), effective_batch=$effective_batch, dim=$(args.dim), n_layers=$(args.n_layers), lr=$(args.lr), save_every=$(args.save_every), prefetch=$(args.prefetch)$(args.decode_every > 0 ? ", decode_every=$(args.decode_every) on fixed val" : ""))")

    # Prefetched batch producer: fills channel so GPU doesn't wait on data
    batch_channel = nothing
    if args.prefetch > 0
        batch_channel = Channel{Any}(args.prefetch)
        if loaded === nothing
            put!(batch_channel, batch)   # use the batch we already generated for n_bins
        end
        @async begin
            for _ in (loaded === nothing ? (step_start + 1) : step_start):args.steps
                put!(batch_channel, generate_batch(cfg, args.batch_size; rng))
            end
            close(batch_channel)
        end
    end

    t0 = time()
    grads_accum = nothing
    accum_count = 0
    loss_sum = 0.0f0

    for step in step_start:args.steps
        batch = args.prefetch > 0 ? take!(batch_channel) : generate_batch(cfg, args.batch_size; rng)
        spec, decoder_input, decoder_target, station_mask, station_ids = prepare_training_batch(batch)
        spec = device(spec)
        decoder_input = device(decoder_input)
        decoder_target = device(decoder_target)
        station_mask = device(station_mask)
        station_ids = device(station_ids)

        args.gpu && CUDA.synchronize()
        t_step = time()
        result = Flux.withgradient(model) do m
            train_step(m, spec, decoder_input, decoder_target, station_mask, station_ids) / Float32(args.accum_steps)
        end
        args.gpu && CUDA.synchronize()
        loss_sum += result.val * args.accum_steps

        if grads_accum === nothing
            grads_accum = result.grad[1]
        else
            grads_accum = Flux.fmap((a, b) -> a isa AbstractArray ? a .+ b : (a isa Number && b isa Number ? a + b : a), grads_accum, result.grad[1])
        end
        accum_count += 1

        if accum_count == args.accum_steps
            Flux.update!(opt, model, grads_accum)
            grads_accum = nothing
            accum_count = 0
            loss_avg = loss_sum / args.accum_steps
            loss_sum = 0.0f0
            if step % 50 == 0
                elapsed_total = time() - t0
                steps_per_sec = (step - step_start + 1) / max(elapsed_total, 1e-9)
                println("  step $step  loss = $(round(loss_avg; digits=4))  ($(round(steps_per_sec; digits=2)) steps/s)")
            end
            if step % args.save_every == 0
                save_checkpoint(checkpoint_path, step, n_bins, model, opt)
            end
            if args.decode_every > 0 && step % args.decode_every == 0
                println("  [decode @ step $step (fixed val)]")
                run_decode_test(model, val_batch, device; n_stations=n_val_stations)
            end
        end
    end

    # Final update if we had a partial accumulation
    if accum_count > 0
        Flux.update!(opt, model, grads_accum)
    end

    # Save final checkpoint if we didn't already save at the last step (e.g. steps=60, save_every=30 → already saved at 60)
    if args.steps >= step_start && args.steps % args.save_every != 0
        save_checkpoint(checkpoint_path, args.steps, n_bins, model, opt)
    end

    # Final decode test on fixed validation (same as during training)
    println("\nDecode test (fixed validation spectrogram, $n_val_stations stations):")
    run_decode_test(model, val_batch, device; n_stations=n_val_stations)
end

main()
