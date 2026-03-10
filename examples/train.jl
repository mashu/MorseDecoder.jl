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
using Optimisers: adjust!
using ParameterSchedulers: OneCycle
using Random
using JLD2
using ChainRulesCore
using UnicodePlots

# Multi-step gradient accumulation: sum grads over N steps then one optimizer update.
# Uses ChainRulesCore.add!! for in-place accumulation when possible (AD-agnostic).
# Not the same as Zygote.checkpointed (that is gradient checkpointing for memory).
accum_grad(a::AbstractArray{T,N}, b::AbstractArray{T,N}) where {T,N} = add!!(a, b)
accum_grad(a::N, b::N) where N<:Number = add!!(a, b)
accum_grad(a, b) = a  # fallback for Nothing / other leaf types

"""Total number of trainable parameters in the model (for capacity logging)."""
function count_parameters(model)
    flat, _ = Flux.destructure(model)
    length(flat)
end

"""Encoder dropout for this step: 0 until schedule_start, then encoder_dropout_max. Use schedule_start=0 to disable."""
function effective_encoder_dropout(step::Int, args)
    args.encoder_dropout_schedule_start <= 0 && return Float64(args.encoder_dropout)
    step >= args.encoder_dropout_schedule_start ? Float64(args.encoder_dropout_max) : 0.0
end

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
        help = "Number of batches to pre-generate and queue (0 = no prefetch). One producer task fills the queue using Threads.nthreads() inside generate_batch; use -t N for N>1."
        arg_type = Int
        default = 256
        "--lr"
        help = "Peak learning rate (with --warmup-fraction > 0, schedule goes startval -> lr -> endval)"
        arg_type = Float32
        default = 3f-4
        "--warmup-fraction"
        help = "Fraction of steps for LR warmup (0 = constant LR). Default 0.1, capped at 2000 steps. One-cycle then cosine decay to lr/100."
        arg_type = Float64
        default = 0.1
        "--decode-every"
        help = "Run decode on fixed val every N steps (0 = only at end)"
        arg_type = Int
        default = 500
        "--dim"
        help = "Model dimension (encoder and decoder)"
        arg_type = Int
        default = 256
        "--n-layers"
        help = "Number of encoder/decoder layers (min 5 for RoPE)"
        arg_type = Int
        default = 8
        "--n-heads"
        help = "Number of attention heads"
        arg_type = Int
        default = 4
        "--teacher-forcing-prob"
        help = "Prob of using ground truth as decoder input (1.0 = standard teacher forcing; <1.0 = scheduled sampling with model predictions at that rate)."
        arg_type = Float64
        default = 1.0
        "--encoder-dropout"
        help = "Fixed prob of zeroing encoder output per step (0 = off). Overridden by --encoder-dropout-schedule when used."
        arg_type = Float64
        default = 0.0
        "--encoder-dropout-schedule-start"
        help = "Step at which to start gradual encoder dropout (0 = disabled; then encoder dropout ramps to --encoder-dropout-max)."
        arg_type = Int
        default = 5000
        "--encoder-dropout-max"
        help = "Max encoder dropout when using schedule (e.g. 0.05 after 5K steps)."
        arg_type = Float64
        default = 0.05
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
      lr = parsed["lr"], warmup_fraction = parsed["warmup-fraction"], decode_every = parsed["decode-every"],
      dim = parsed["dim"], n_layers, n_heads = parsed["n-heads"],
      teacher_forcing_prob = parsed["teacher-forcing-prob"], encoder_dropout = parsed["encoder-dropout"],
      encoder_dropout_schedule_start = parsed["encoder-dropout-schedule-start"],
      encoder_dropout_max = parsed["encoder-dropout-max"],
      benchmark = parsed["benchmark"])
end

function build_model(n_bins::Int; dim=256, n_heads=4, n_layers=8)
    encoder = SpectrogramEncoder(n_bins, dim, n_heads, n_layers)
    decoder = SpectrogramDecoder(VOCAB_SIZE, dim, n_heads, n_layers)
    SpectrogramEncoderDecoder(encoder, decoder)
end

"""Run decode on one spectrogram (single stream with speaker tokens); print truth vs decoded."""
function run_decode_test(model, batch, device; max_len::Int=128)
    run_decode_one(model, batch, device, max_len)
end

"""Run decode on each of several batches."""
function run_decode_test(model, batches::AbstractVector, device; max_len::Int=128)
    for (i, batch) in enumerate(batches)
        @info "val sample" sample=i
        run_decode_one(model, batch, device, max_len)
    end
end

function run_decode_one(model, batch, device, max_len)
    test_spec = batch.spectrogram[:, 1:1, :]
    test_spec = device(test_spec)
    ids = decode_autoregressive(model, test_spec; max_len, to_device=device)
    ids_cpu = cpu(ids)
    truth_ids = batch.targets[1, :]
    truth_s = token_ids_to_string_with_speakers(truth_ids)
    seq = [ids_cpu[i, 1] for i in 1:size(ids_cpu, 1)]
    decode_s = token_ids_to_string_with_speakers(seq)
    @info "decode" truth=(isempty(truth_s) ? "(empty)" : truth_s) decode=(isempty(decode_s) ? "(empty)" : decode_s)
end

"""Turn token ids into string; stop at EOS/PAD/0. Speaker tokens shown as [1]..[6]; chars as-is."""
function token_ids_to_string_with_speakers(ids::AbstractVector{<:Integer})
    buf = Char[]
    for i in ids
        (i == EOS_TOKEN_IDX || i == PAD_TOKEN_IDX || i == 0) && break
        i == START_TOKEN_IDX && continue
        if is_speaker_token(i)
            append!(buf, ['[', Char('0' + (i - SPEAKER_1_IDX + 1)), ']'])
        elseif 1 <= i <= NUM_CHARS
            push!(buf, IDX_TO_CHAR[i])
        end
    end
    String(buf)
end

"""Like token_ids_to_string_with_speakers but strips speaker tokens (plain text only)."""
function token_ids_to_string(ids::AbstractVector{<:Integer})
    buf = Char[]
    for i in ids
        (i == EOS_TOKEN_IDX || i == PAD_TOKEN_IDX || i == 0) && break
        i == START_TOKEN_IDX && continue
        is_speaker_token(i) && continue
        1 <= i <= NUM_CHARS && push!(buf, IDX_TO_CHAR[i])
    end
    String(buf)
end

function save_checkpoint(path::String, step::Int, n_bins::Int, model, opt; dim::Int, n_layers::Int, n_heads::Int)
    mkpath(dirname(path))
    model_cpu = cpu(model)
    opt_cpu = cpu(opt)
    model_state = Flux.state(model_cpu)
    jldsave(path; step, n_bins, dim, n_layers, n_heads,
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
        batch = generate_batch_fast(cfg, args.batch_size; rng)
        spec, decoder_input, decoder_target = prepare_training_batch(batch)
        spec = device(spec)
        decoder_input = device(decoder_input)
        decoder_target = device(decoder_target)
        result = Flux.withgradient(model) do m
            train_step(m, spec, decoder_input, decoder_target;
                encoder_dropout = 0.0, rng = rng) / args.accum_steps
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
        batch = generate_batch_fast(cfg, args.batch_size; rng)
        spec, decoder_input, decoder_target = prepare_training_batch(batch)
        push!(t_data, time() - t0)

        t0 = time()
        spec = device(spec)
        decoder_input = device(decoder_input)
        decoder_target = device(decoder_target)
        push!(t_transfer, time() - t0)

        t0 = time()
        result = Flux.withgradient(model) do m
            train_step(m, spec, decoder_input, decoder_target;
                encoder_dropout = 0.0, rng = rng) / args.accum_steps
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
        model_state = f["model_state"]
        optimiser_state = f["optimiser_state"]
        dim = haskey(f, "dim") ? f["dim"] : nothing
        n_layers = haskey(f, "n_layers") ? f["n_layers"] : nothing
        n_heads = haskey(f, "n_heads") ? f["n_heads"] : nothing
        (; step, n_bins, model_state, optimiser_state, dim, n_layers, n_heads)
    end
end

function main()
    args = parse_commandline()
    rng = MersenneTwister(42)
    device = args.gpu ? gpu : cpu
    checkpoint_path = joinpath(args.checkpoint_dir, "checkpoint_latest.jld2")

    # Data: 1–3 stations. Spectrogram is band-limited to 100–900 Hz (Morse is 200–800 Hz); we never feed full FFT, only n_bins in that range. Cap at 512 frames for GPU memory.
    cfg = SamplerConfig(; n_stations_range=1:3, spec=SpectrogramConfig(; freq_lo=100f0, freq_hi=900f0, max_frames=512))
    batch = generate_batch_fast(cfg, args.batch_size; rng)
    n_bins = size(batch.spectrogram, 1)

    loaded = load_checkpoint(checkpoint_path)
    step_start = 1
    if loaded !== nothing
        d = loaded
        dim = something(get(d, :dim, nothing), args.dim)
        n_layers = something(get(d, :n_layers, nothing), args.n_layers)
        n_heads = something(get(d, :n_heads, nothing), args.n_heads)
        model = build_model(d.n_bins; dim, n_heads, n_layers)
        try
            Flux.loadmodel!(model, d.model_state)
            step_start = d.step + 1
            n_bins = d.n_bins
            opt = d.optimiser_state
            if args.gpu
                model = device(model)
                opt = device(opt)
            end
            @info "Resumed from checkpoint" path=checkpoint_path from_step=d.step continuing_from=step_start dim=dim n_layers=n_layers n_heads=n_heads
        catch e
            @warn "Checkpoint incompatible (e.g. old architecture); starting fresh" path=checkpoint_path exception=(e isa Exception ? e : nothing)
            model = build_model(n_bins; dim=args.dim, n_heads=args.n_heads, n_layers=args.n_layers)
            if args.gpu
                model = device(model)
            end
            opt = Flux.setup(Adam(args.lr), model)
        end
    else
        model = build_model(n_bins; dim=args.dim, n_heads=args.n_heads, n_layers=args.n_layers)
        if args.gpu
            model = device(model)
        end
        opt = Flux.setup(Adam(args.lr), model)
    end

    # LR schedule: one-cycle (warmup then cosine decay) or constant. Good defaults: warmup capped at 2000 steps, non-zero start/end LR.
    lr_schedule = if args.warmup_fraction > 0
        warmup_steps = min(ceil(Int, args.steps * args.warmup_fraction), 2000)
        percent_start = warmup_steps / args.steps
        startval = max(1f-8, args.lr / 25f0)
        endval = max(1f-9, args.lr / 100f0)
        @info "LR schedule" warmup_steps peak_lr=args.lr startval endval
        OneCycle(args.steps, args.lr; startval, endval, percent_start)
    else
        t -> args.lr
    end

    # Plot LR schedule so we can inspect it before training
    steps_remaining = args.steps - step_start + 1
    if steps_remaining > 0
        n_sample = min(400, steps_remaining)
        steps_sample = round.(Int, range(step_start, args.steps; length=n_sample))
        lr_vals = Float64.(lr_schedule.(steps_sample))
        plt = lineplot(steps_sample, lr_vals; title="LR schedule", xlabel="step", ylabel="learning rate", width=60, height=15)
        println(plt)
    end

    effective_batch = args.batch_size * args.accum_steps
    # Several fixed validation batches (different seeds) so we see if decode varies with input
    val_batches = [generate_batch_fast(cfg, 1; rng=MersenneTwister(s)) for s in [123, 456, 789]]

    n_params = count_parameters(model)
    @info "Training" steps=args.steps device=(args.gpu ? "GPU" : "CPU") batch=args.batch_size accum=args.accum_steps effective_batch=effective_batch dim=args.dim n_layers=args.n_layers n_heads=args.n_heads n_params=n_params lr=args.lr warmup_fraction=args.warmup_fraction save_every=args.save_every prefetch=args.prefetch prefetch_threads=Threads.nthreads() decode_every=args.decode_every teacher_forcing=args.teacher_forcing_prob encoder_dropout=args.encoder_dropout encoder_dropout_schedule_start=args.encoder_dropout_schedule_start encoder_dropout_max=args.encoder_dropout_max

    if args.benchmark > 0
        run_benchmark(args, model, opt, cfg, device, n_bins)
        return
    end

    # Prefetched batch producer: one async task fills the channel; each generate_batch uses
    # Threads.nthreads() (run with -t N). Pre-fill queue so GPU has a buffer before we start.
    batch_channel = nothing
    if args.prefetch > 0
        batch_channel = Channel{Any}(args.prefetch)
        sync_prefill = Channel{Nothing}(1)
        @async begin
            prefill = min(args.prefetch, args.steps - step_start + 1)
            for _ in 1:prefill
                put!(batch_channel, generate_batch_fast(cfg, args.batch_size; rng))
            end
            put!(sync_prefill, nothing)
            remaining = args.steps - step_start + 1 - prefill
            for _ in 1:remaining
                put!(batch_channel, generate_batch_fast(cfg, args.batch_size; rng))
            end
            close(batch_channel)
        end
        take!(sync_prefill)
        close(sync_prefill)
    end

    t0 = time()
    grads_accum = nothing
    accum_count = 0
    loss_sum = 0.0f0

    for step in step_start:args.steps
        batch = args.prefetch > 0 ? take!(batch_channel) : generate_batch_fast(cfg, args.batch_size; rng)
        spec, decoder_input, decoder_target = prepare_training_batch(batch)
        spec = device(spec)
        decoder_input = device(decoder_input)
        decoder_target = device(decoder_target)

        enc_drop = effective_encoder_dropout(step, args)
        result = Flux.withgradient(model) do m
            train_step(m, spec, decoder_input, decoder_target;
                encoder_dropout = enc_drop, rng = rng) / args.accum_steps
        end
        loss_sum += result.val * args.accum_steps

        if grads_accum === nothing
            grads_accum = result.grad[1]
        else
            grads_accum = Flux.fmap(accum_grad, grads_accum, result.grad[1])
        end
        accum_count += 1

        if accum_count == args.accum_steps
            eta = Float32(lr_schedule(min(step, args.steps)))
            adjust!(opt, eta)
            Flux.update!(opt, model, grads_accum)
            grads_accum = nothing
            accum_count = 0
            loss_avg = loss_sum / args.accum_steps
            loss_sum = 0.0f0
            if step % 50 == 0
                elapsed_total = time() - t0
                steps_per_sec = (step - step_start + 1) / max(elapsed_total, 1e-9)
                eta = Float64(lr_schedule(min(step, args.steps)))
                @info "step" step=step loss=round(loss_avg; digits=4) lr=eta steps_per_sec=round(steps_per_sec; digits=2)
            end
            if step % args.save_every == 0
                save_checkpoint(checkpoint_path, step, n_bins, model, opt; dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads)
            end
            if args.decode_every > 0 && step % args.decode_every == 0
                @info "decode" step=step n_samples=3
                run_decode_test(model, val_batches, device)
            end
        end
    end

    # Final update if we had a partial accumulation
    if accum_count > 0
        eta = Float32(lr_schedule(args.steps))
        adjust!(opt, eta)
        Flux.update!(opt, model, grads_accum)
    end

    # Save final checkpoint if we didn't already save at the last step (e.g. steps=60, save_every=30 → already saved at 60)
    if args.steps >= step_start && args.steps % args.save_every != 0
        save_checkpoint(checkpoint_path, args.steps, n_bins, model, opt; dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads)
    end

    # Final decode test on fixed validation (same as during training)
    @info "Decode test" n_spectrograms=3
    run_decode_test(model, val_batches, device)
end

main()
