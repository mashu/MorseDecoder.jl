#!/usr/bin/env julia
# Train the spectrogram encoder–decoder. Run from repo root:
#   julia --project=. examples/train.jl [options]
# Use --help for full list and defaults.
# Load CUDA/cuDNN before Flux so Flux uses the GPU backend.
#
# Benchmarking: julia -t 4 --project=. examples/train.jl --gpu --benchmark 50
#   Reports timing breakdown (data / transfer / forward+backward / accum+update).
#   Use -t N (N>1) so batch generation runs in parallel and keeps GPU fed.

using CUDA
using cuDNN
using ArgParse
using Logging
using MorseDecoder
using Flux
using Random
using JLD2
using ChainRulesCore

# Multi-step gradient accumulation: sum grads over N steps then one optimizer update.
# Uses ChainRulesCore.add!! for in-place accumulation when possible (AD-agnostic).
# Not the same as Zygote.checkpointed (that is gradient checkpointing for memory).
accum_grad(a::AbstractArray{T,N}, b::AbstractArray{T,N}) where {T,N} = add!!(a, b)
accum_grad(a::N, b::N) where N<:Number = add!!(a, b)
accum_grad(a, b) = a  # fallback for Nothing / other leaf types

function parse_commandline()
    s = ArgParseSettings(
        description = "Train spectrogram encoder–decoder for multi-station Morse. Checkpoints (model + optimiser) saved in checkpoint-dir; resume by re-running with same dir.",
        version = "train.jl 0.1",
        add_version = true,
    )
    @add_arg_table! s begin
        "--gpu"
        help = "Use GPU (CUDA) for training and decode"
        action = :store_true
        "--steps"
        help = "Number of training steps"
        arg_type = Int
        default = 100_000
        "--batch"
        help = "Batch size (default 64 with --gpu; use 8 for CPU)"
        arg_type = Int
        default = 64
        "--accum"
        help = "Gradient accumulation steps (effective batch = batch × accum)"
        arg_type = Int
        default = 1
        "--checkpoint-dir"
        help = "Directory for checkpoint_latest.jld2"
        arg_type = String
        default = "checkpoints"
        "--save-every"
        help = "Save checkpoint every N steps"
        arg_type = Int
        default = 500
        "--prefetch"
        help = "Number of batches to pre-generate (0 = no prefetch)"
        arg_type = Int
        default = 128
        "--lr"
        help = "Learning rate"
        arg_type = Float32
        default = 1f-5
        "--decode-every"
        help = "Run decode on fixed val every N steps (0 = only at end)"
        arg_type = Int
        default = 500
        "--dim"
        help = "Model dimension (encoder and decoder)"
        arg_type = Int
        default = 64
        "--n-layers"
        help = "Number of encoder/decoder layers (min 5 for RoPE)"
        arg_type = Int
        default = 6
        "--n-heads"
        help = "Number of attention heads"
        arg_type = Int
        default = 4
        "--teacher-forcing-prob"
        help = "Prob of using ground truth as decoder input (1.0 = always; 0.9 = 90% GT, 10% random). Reduces exposure bias."
        arg_type = Float64
        default = 0.9
        "--benchmark"
        help = "If >0, run N steps with timing breakdown then exit (no checkpoint/decode)"
        arg_type = Int
        default = 0
    end
    parsed = ArgParse.parse_args(ARGS, s)
    # CPU: default batch 8 if user did not pass --batch (parser default is 64)
    batch_size = parsed["gpu"] ? parsed["batch"] : (parsed["batch"] == 64 ? 8 : parsed["batch"])
    n_layers = max(5, parsed["n-layers"])
    (; gpu = parsed["gpu"], steps = parsed["steps"], batch_size,
      accum_steps = parsed["accum"], checkpoint_dir = parsed["checkpoint-dir"],
      save_every = parsed["save-every"], prefetch = parsed["prefetch"],
      lr = parsed["lr"], decode_every = parsed["decode-every"],
      dim = parsed["dim"], n_layers, n_heads = parsed["n-heads"],
      teacher_forcing_prob = parsed["teacher-forcing-prob"], benchmark = parsed["benchmark"])
end

function build_model(n_bins::Int, chunk_frames::Int; dim=128, n_heads=4, n_layers=2, max_stations::Int=5)
    token_dim = n_bins * chunk_frames
    encoder = SpectrogramEncoder(token_dim, dim, n_heads, n_layers, chunk_frames)
    decoder = SpectrogramDecoder(VOCAB_SIZE, dim, n_heads, n_layers; max_stations)
    SpectrogramEncoderDecoder(encoder, decoder)
end

const CHUNK_FRAMES = 4

"""Run decode on one spectrogram, n_stations outputs; print ground truth and decoded text for comparison."""
function run_decode_test(model, batch, device; n_stations::Int=2, max_len::Int=32)
    _run_decode_one(model, batch, device, n_stations, max_len)
end

"""Run decode on each of several batches (e.g. different val seeds) to see if output varies with input."""
function run_decode_test(model, batches::AbstractVector, device; n_stations::Int=2, max_len::Int=32)
    for (i, batch) in enumerate(batches)
        @info "val sample" sample=i
        _run_decode_one(model, batch, device, n_stations, max_len)
    end
end

function _run_decode_one(model, batch, device, n_stations, max_len)
    test_spec = batch.spectrogram[:, 1:1, :]
    test_spec = device(test_spec)
    ids = decode_autoregressive(model, test_spec, n_stations; max_len, to_device=device)
    ids_cpu = cpu(ids)
    n_stations_actual = min(n_stations, size(batch.targets, 2))
    for k in 1:n_stations_actual
        truth_ids = batch.targets[1, k, :]
        truth_s = token_ids_to_string(truth_ids)
        seq = [ids_cpu[i, k] for i in 1:size(ids_cpu, 1)]
        decode_s = token_ids_to_string(seq)
        @info "station" station=k truth=(isempty(truth_s) ? "(empty)" : truth_s) decode=(isempty(decode_s) ? "(empty)" : decode_s)
    end
end

"""Turn decoded token id sequence into a string: chars 1:NUM_CHARS, BLANK→space, stop at PAD/START/0."""
function token_ids_to_string(ids::AbstractVector{<:Integer})
    buf = Char[]
    for i in ids
        (i == PAD_TOKEN_IDX || i == 0) && break
        i == START_TOKEN_IDX && continue
        if 1 <= i <= NUM_CHARS
            push!(buf, IDX_TO_CHAR[i])
        elseif i == BLANK_TOKEN
            push!(buf, ' ')
        end
    end
    String(buf)
end

function save_checkpoint(path::String, step::Int, n_bins::Int, model, opt; dim::Int, n_layers::Int, n_heads::Int)
    mkpath(dirname(path))
    model_cpu = cpu(model)
    opt_cpu = cpu(opt)
    model_state = Flux.state(model_cpu)
    jldsave(path; step, n_bins, chunk_frames=CHUNK_FRAMES, dim, n_layers, n_heads,
            model_state, optimiser_state=opt_cpu)
    @info "checkpoint saved" step=step path=path
end

"""Run N training steps with timing breakdown (data, transfer, forward+backward, accum+update). No prefetch so data cost is explicit. Use --gpu --benchmark 50 for GPU timing."""
function run_benchmark(args, model, opt, cfg, device, n_bins)
    n_steps = args.benchmark
    warmup = min(5, max(1, n_steps ÷ 4))
    @info "Benchmark run" n_steps=n_steps warmup=warmup device=(args.gpu ? "GPU" : "CPU") batch=args.batch_size threads=Threads.nthreads()
    rng = MersenneTwister(123)
    t_data = Float64[]
    t_transfer = Float64[]
    t_fwbw = Float64[]
    t_accum = Float64[]
    sizehint!(t_data, n_steps)
    sizehint!(t_transfer, n_steps)
    sizehint!(t_fwbw, n_steps)
    sizehint!(t_accum, n_steps)

    grads_accum = nothing
    accum_count = 0

    # Warmup
    for _ in 1:warmup
        batch = generate_batch(cfg, args.batch_size; rng)
        spec, decoder_input, decoder_target, station_mask, station_ids = prepare_training_batch(batch)
        spec = device(spec)
        decoder_input = device(decoder_input)
        decoder_target = device(decoder_target)
        station_mask = device(station_mask)
        station_ids = device(station_ids)
        if args.teacher_forcing_prob < 1
            noise_prob = 1.0 - args.teacher_forcing_prob
            L, Bk = size(decoder_input)
            noise_mask = rand(rng, L, Bk) .< noise_prob
            noise_mask[1, :] .= false
            random_tokens = rand(rng, 1:VOCAB_SIZE, L, Bk)
            noise_mask = device(noise_mask)
            random_tokens = device(random_tokens)
            decoder_input = ifelse.(noise_mask, random_tokens, decoder_input)
        end
        result = Flux.withgradient(model) do m
            train_step(m, spec, decoder_input, decoder_target, station_mask, station_ids) / args.accum_steps
        end
        if grads_accum === nothing
            grads_accum = result.grad[1]
        else
            grads_accum = Flux.fmap(accum_grad, grads_accum, result.grad[1])
        end
        accum_count += 1
        if accum_count == args.accum_steps
            Flux.update!(opt, model, grads_accum)
            grads_accum = nothing
            accum_count = 0
        end
    end
    grads_accum = nothing
    accum_count = 0
    if args.gpu
        CUDA.synchronize()
    end

    # Timed run
    for _ in 1:n_steps
        t0 = time()
        batch = generate_batch(cfg, args.batch_size; rng)
        spec, decoder_input, decoder_target, station_mask, station_ids = prepare_training_batch(batch)
        push!(t_data, time() - t0)

        t0 = time()
        spec = device(spec)
        decoder_input = device(decoder_input)
        decoder_target = device(decoder_target)
        station_mask = device(station_mask)
        station_ids = device(station_ids)
        if args.teacher_forcing_prob < 1
            noise_prob = 1.0 - args.teacher_forcing_prob
            L, Bk = size(decoder_input)
            noise_mask = rand(rng, L, Bk) .< noise_prob
            noise_mask[1, :] .= false
            random_tokens = rand(rng, 1:VOCAB_SIZE, L, Bk)
            random_tokens = device(random_tokens)
            noise_mask = device(noise_mask)
            decoder_input = ifelse.(noise_mask, random_tokens, decoder_input)
        end
        push!(t_transfer, time() - t0)

        t0 = time()
        result = Flux.withgradient(model) do m
            train_step(m, spec, decoder_input, decoder_target, station_mask, station_ids) / args.accum_steps
        end
        push!(t_fwbw, time() - t0)

        t0 = time()
        if grads_accum === nothing
            grads_accum = result.grad[1]
        else
            grads_accum = Flux.fmap(accum_grad, grads_accum, result.grad[1])
        end
        accum_count += 1
        if accum_count == args.accum_steps
            Flux.update!(opt, model, grads_accum)
            grads_accum = nothing
            accum_count = 0
        end
        push!(t_accum, time() - t0)
    end

    if args.gpu
        CUDA.synchronize()
    end

    sum_data = sum(t_data) * 1000
    sum_transfer = sum(t_transfer) * 1000
    sum_fwbw = sum(t_fwbw) * 1000
    sum_accum = sum(t_accum) * 1000
    total_ms = sum_data + sum_transfer + sum_fwbw + sum_accum
    steps_per_sec = n_steps / (total_ms / 1000)

    @info "Benchmark" steps=n_steps steps_per_sec=round(steps_per_sec; digits=2)
    @info "Timing (ms)" data=round(sum_data; digits=1) data_pct=round(100 * sum_data / total_ms; digits=1) transfer=round(sum_transfer; digits=1) transfer_pct=round(100 * sum_transfer / total_ms; digits=1) forward_backward=round(sum_fwbw; digits=1) fwbw_pct=round(100 * sum_fwbw / total_ms; digits=1) accum_update=round(sum_accum; digits=1) accum_pct=round(100 * sum_accum / total_ms; digits=1)
end

function load_checkpoint(path::String)
    isfile(path) || return nothing
    jldopen(path, "r") do f
        step = f["step"]
        n_bins = f["n_bins"]
        chunk_frames = f["chunk_frames"]
        model_state = f["model_state"]
        optimiser_state = f["optimiser_state"]
        # Architecture: use saved so resume matches; old checkpoints may not have these
        dim = haskey(f, "dim") ? f["dim"] : nothing
        n_layers = haskey(f, "n_layers") ? f["n_layers"] : nothing
        n_heads = haskey(f, "n_heads") ? f["n_heads"] : nothing
        (; step, n_bins, chunk_frames, model_state, optimiser_state, dim, n_layers, n_heads)
    end
end

function main()
    args = parse_commandline()
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
        d = loaded
        step_start = d.step + 1
        # Use saved architecture so weights match; fall back to args for old checkpoints
        dim = something(get(d, :dim, nothing), args.dim)
        n_layers = something(get(d, :n_layers, nothing), args.n_layers)
        n_heads = something(get(d, :n_heads, nothing), args.n_heads)
        n_bins = d.n_bins
        chunk_frames = d.chunk_frames
        model_state = d.model_state
        optimiser_state = d.optimiser_state
        model = build_model(n_bins, chunk_frames; dim, n_heads, n_layers)
        Flux.loadmodel!(model, model_state)
        opt = optimiser_state
        if args.gpu
            model = device(model)
            opt = device(opt)
        end
        @info "Resumed from checkpoint" path=checkpoint_path from_step=d.step continuing_from=step_start dim=dim n_layers=n_layers n_heads=n_heads
    else
        model = build_model(n_bins, CHUNK_FRAMES; dim=args.dim, n_heads=args.n_heads, n_layers=args.n_layers)
        if args.gpu
            model = device(model)
        end
        opt = Flux.setup(Adam(args.lr), model)
    end

    effective_batch = args.batch_size * args.accum_steps
    # Several fixed validation batches (different seeds) so we see if decode varies with input
    val_batches = [generate_batch(cfg, 1; rng=MersenneTwister(s)) for s in [123, 456, 789]]
    n_val_stations = min(2, minimum(maximum(b.n_stations) for b in val_batches))

    @info "Training" steps=args.steps device=(args.gpu ? "GPU" : "CPU") batch=args.batch_size accum=args.accum_steps effective_batch=effective_batch dim=args.dim n_layers=args.n_layers lr=args.lr save_every=args.save_every prefetch=args.prefetch decode_every=args.decode_every teacher_forcing=args.teacher_forcing_prob

    if args.benchmark > 0
        run_benchmark(args, model, opt, cfg, device, n_bins)
        return
    end

    # Prefetched batch producer: fills channel so GPU doesn't wait on data.
    # With JULIA_NUM_THREADS>1, generate_batch uses parallel sample generation to keep GPU fed.
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

        # Scheduled-sampling style: with prob (1 - teacher_forcing_prob) replace decoder input (except START) with random token so model learns to use encoder when prefix is wrong
        if args.teacher_forcing_prob < 1
            noise_prob = 1.0 - args.teacher_forcing_prob
            L, Bk = size(decoder_input)
            noise_mask = rand(rng, L, Bk) .< noise_prob
            noise_mask[1, :] .= false   # keep START
            random_tokens = rand(rng, 1:VOCAB_SIZE, L, Bk)
            random_tokens = device(random_tokens)
            noise_mask = device(noise_mask)
            decoder_input = ifelse.(noise_mask, random_tokens, decoder_input)
        end

        t_step = time()
        result = Flux.withgradient(model) do m
            train_step(m, spec, decoder_input, decoder_target, station_mask, station_ids) / args.accum_steps
        end
        loss_sum += result.val * args.accum_steps

        if grads_accum === nothing
            grads_accum = result.grad[1]
        else
            grads_accum = Flux.fmap(accum_grad, grads_accum, result.grad[1])
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
                @info "step" step=step loss=round(loss_avg; digits=4) steps_per_sec=round(steps_per_sec; digits=2)
            end
            if step % args.save_every == 0
                save_checkpoint(checkpoint_path, step, n_bins, model, opt; dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads)
            end
            if args.decode_every > 0 && step % args.decode_every == 0
                @info "decode" step=step n_samples=3
                run_decode_test(model, val_batches, device; n_stations=n_val_stations)
            end
        end
    end

    # Final update if we had a partial accumulation
    if accum_count > 0
        Flux.update!(opt, model, grads_accum)
    end

    # Save final checkpoint if we didn't already save at the last step (e.g. steps=60, save_every=30 → already saved at 60)
    if args.steps >= step_start && args.steps % args.save_every != 0
        save_checkpoint(checkpoint_path, args.steps, n_bins, model, opt; dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads)
    end

    # Final decode test on fixed validation (same as during training)
    @info "Decode test" n_spectrograms=3 n_stations=n_val_stations
    run_decode_test(model, val_batches, device; n_stations=n_val_stations)
end

main()
