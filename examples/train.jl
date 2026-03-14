#!/usr/bin/env julia
# Train the spectrogram encoder–decoder. Run from repo root:
#   julia --project=. examples/train.jl [options]
# Use --help for full list and defaults.
# With --gpu, CUDA/cuDNN are loaded (before Flux) for GPU backend.
#
# Benchmarking: julia -t 4 --project=. examples/train.jl --gpu --benchmark 50 [--batch 32]
#   Use -t N (N>1) so batch generation runs in parallel. If OOM, use --batch 32.
# GPU kernel profiling: nsys launch julia -t 4 --project=. examples/profile_gpu.jl --profile --batch 32 --steps 5
#   Then open report.qdrep in nsight-sys to see which kernels dominate.

using ArgParse
using Logging
# Load CUDA/cuDNN before Flux when --gpu (must be at top level)
if "--gpu" in ARGS
    using CUDA
    using cuDNN
end
using MorseDecoder
using Flux
using Optimisers: adjust!
using ParameterSchedulers: OneCycle
using Random
using JLD2
using ChainRulesCore
using UnicodePlots
using CannotWaitForTheseOptimisers

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
        default = Float32(1.5f-4)
        "--warmup-fraction"
        help = "Fraction of steps for LR warmup when --warmup-steps not set (0 = constant LR). One-cycle then cosine decay to lr/100."
        arg_type = Float64
        default = 0.01
        "--warmup-steps"
        help = "Fixed number of warmup steps (overrides warmup-fraction when > 0). Use e.g. 500 to avoid long percentage-based warmup."
        arg_type = Int
        default = 0
        "--decode-every"
        help = "Run decode on fixed val every N steps (0 = only at end)"
        arg_type = Int
        default = 500
        "--dim"
        help = "Model dimension (decoder; also encoder when --encoder-dim not set)"
        arg_type = Int
        default = 384
        "--encoder-dim"
        help = "Encoder dimension (default: same as dim). Larger encoder_dim gives CTC/encoder more capacity; use projection to decoder dim."
        arg_type = Int
        default = 0
        "--encoder-layers"
        help = "Number of encoder (self-attention) layers (min 5 for RoPE). Whisper-tiny uses 4."
        arg_type = Int
        default = 6
        "--decoder-layers"
        help = "Decoder self-attn layers (0 = cross-only decoder; each pairs with one cross-attn, interleaved)"
        arg_type = Int
        default = 4
        "--cross-layers"
        help = "Cross-attention layers (0 = same as decoder-layers; 6 default = 4 paired + 2 cross-only)"
        arg_type = Int
        default = 6
        "--n-heads"
        help = "Number of attention heads"
        arg_type = Int
        default = 6
        "--decoder-input-dropout"
        help = "Dropout on decoder embeddings (e.g. 0.1) to encourage cross-attention use"
        arg_type = Float32
        default = Float32(0.1)
        "--self-attn-residual-scale"
        help = "Scale for decoder self-attention residual (1 = normal, Whisper-style; <1 = rely more on encoder)"
        arg_type = Float32
        default = Float32(1.0)
        "--max-frames"
        help = "Cap spectrogram time frames per sample (default 512). Fewer = less GPU memory and faster; need enough for longest transcript. Frontend already does 2× downsampling (~4 encoder frames/dot at 50 WPM)."
        arg_type = Int
        default = 512
        "--benchmark"
        help = "If >0, run N steps with timing breakdown then exit (no checkpoint/decode)"
        arg_type = Int
        default = 0
        "--ctc-weight"
        help = "Weight for CTC loss (0 = no CTC). Light nudge (e.g. 0.05–0.1) keeps attention dominant; default 0.1"
        arg_type = Float32
        default = Float32(0.1)
        "--label-smoothing"
        help = "Label smoothing for decoder CE (0 = none; 0.1 default to reduce overconfident collapse)"
        arg_type = Float32
        default = Float32(0.1)
    end
    parsed = ArgParse.parse_args(ARGS, s)
    # CPU: default batch 8 if user did not pass --batch (parser default is 64)
    batch_size = parsed["gpu"] ? parsed["batch"] : (parsed["batch"] == 64 ? 8 : parsed["batch"])
    encoder_layers = max(5, parsed["encoder-layers"])
    decoder_layers = max(0, min(parsed["decoder-layers"], encoder_layers))
    cross_layers_raw = parsed["cross-layers"]
    cross_layers = (cross_layers_raw == 0 ? (decoder_layers == 0 ? 2 : decoder_layers) : max(decoder_layers, cross_layers_raw))
    ctc_weight = parsed["ctc-weight"]
    encoder_dim_raw = parsed["encoder-dim"]
    encoder_dim = encoder_dim_raw <= 0 ? nothing : encoder_dim_raw  # 0 => use dim for both
    (; gpu = parsed["gpu"], steps = parsed["steps"], batch_size,
      accum_steps = parsed["accum"], checkpoint_dir = parsed["checkpoint-dir"],
      save_every = parsed["save-every"], prefetch = parsed["prefetch"],
      max_frames = parsed["max-frames"],
      lr = parsed["lr"], warmup_fraction = parsed["warmup-fraction"], warmup_steps = parsed["warmup-steps"],
      decode_every = parsed["decode-every"],
      dim = parsed["dim"], encoder_dim, encoder_layers,
      decoder_layers, cross_layers, n_heads = parsed["n-heads"],
      decoder_input_dropout = parsed["decoder-input-dropout"],
      self_attn_residual_scale = parsed["self-attn-residual-scale"],
      benchmark = parsed["benchmark"],
      ctc_weight, label_smoothing = parsed["label-smoothing"])
end

function build_model(n_bins::Int; dim=384, encoder_dim=nothing, n_heads=6, encoder_layers=6, decoder_layers=2, cross_layers=2, decoder_input_dropout=Float32(0.1), self_attn_residual_scale=Float32(1.0))
    enc_dim = something(encoder_dim, dim)  # encoder_dim == dim (default) => single shared dim
    encoder = SpectrogramEncoder(n_bins, enc_dim, n_heads, encoder_layers)
    decoder = SpectrogramDecoder(VOCAB_SIZE, dim, n_heads, decoder_layers;
        n_cross_layers=cross_layers, decoder_input_dropout, self_attn_residual_scale)
    ctc_head = Dense(enc_dim => CTC_VOCAB_SIZE)
    encoder_proj = (enc_dim == dim) ? nothing : Dense(enc_dim => dim)
    SpectrogramEncoderDecoder(encoder, decoder, ctc_head, encoder_proj)
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
    ids_cpu = cpu(ids)  # minimal transfer: token IDs only, for token_ids_to_label (expects CPU indices)
    truth_ids = batch.targets[1, :]
    truth_s = token_ids_to_string_with_speakers(truth_ids)
    seq = [ids_cpu[i, 1] for i in 1:size(ids_cpu, 1)]
    decode_s = token_ids_to_string_with_speakers(seq)
    ctc_seqs = ctc_greedy_decode(model, test_spec; input_lengths=[batch.input_lengths[1]])
    ctc_s = token_ids_to_string_with_speakers(ctc_seqs[1])
    @info "decode" truth=(isempty(truth_s) ? "(empty)" : truth_s) decoder=(isempty(decode_s) ? "(empty)" : decode_s) ctc=(isempty(ctc_s) ? "(empty)" : ctc_s)
end

"""Display token ids as label string (with [S1]..[S6], [TS], [TE])."""
token_ids_to_string_with_speakers(ids::AbstractVector{<:Integer}) = token_ids_to_label(ids)

"""Plain text only (no special tokens)."""
token_ids_to_string(ids::AbstractVector{<:Integer}) = token_ids_to_plain_text(ids)

function save_checkpoint(path::String, step::Int, n_bins::Int, model, opt; dim::Int, encoder_dim=nothing, encoder_layers::Int, n_heads::Int, decoder_layers::Int, cross_layers::Int, decoder_input_dropout::Float32=0.0f0, self_attn_residual_scale::Float32=1.0f0, ctc_weight::Float32=0.0f0)
    mkpath(dirname(path))
    model_cpu = cpu(model)  # checkpoint serialization: save CPU state (no device in file)
    model_state = Flux.state(model_cpu)
    enc_dim_save = something(encoder_dim, dim)  # store encoder_dim for resume (same as dim when not separate)
    # Do not save optimiser_state: Muon (and some other optimisers) contain anonymous functions
    # that JLD2 cannot serialize. On resume we create a fresh optimizer and apply the LR schedule.
    jldsave(path; step, n_bins, dim, encoder_dim=enc_dim_save, encoder_layers, n_heads, decoder_layers, cross_layers, decoder_input_dropout, self_attn_residual_scale,
            ctc_weight, model_state)
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
            train_step(m, spec, decoder_input, decoder_target) / args.accum_steps
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
            train_step(m, spec, decoder_input, decoder_target) / args.accum_steps
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

    data_batches_per_sec = 1000.0 / (sum_data / n_steps)  # batches/sec this benchmark would need from data
    @info "Benchmark" steps=n_steps steps_per_sec=round(steps_per_sec; digits=2) data_must_supply_batches_per_sec=round(data_batches_per_sec; digits=2)
    @info "Timing (ms)" data=round(sum_data; digits=1) data_pct=round(100 * sum_data / total_ms; digits=1) transfer=round(sum_transfer; digits=1) transfer_pct=round(100 * sum_transfer / total_ms; digits=1) forward_backward=round(sum_fwbw; digits=1) fwbw_pct=round(100 * sum_fwbw / total_ms; digits=1) accum_update=round(sum_accum; digits=1) accum_pct=round(100 * sum_accum / total_ms; digits=1)
    if sum_fwbw > 0.7 * total_ms
        @info "Bottleneck" hint="GPU compute (forward+backward) dominates. To utilize GPU more: increase batch size if memory allows, or use gradient checkpointing to free memory for larger batch."
    elseif sum_data > 0.15 * total_ms
        @info "Bottleneck" hint="Data generation is a significant share (benchmark has no prefetch). In training, use -t N and --prefetch to overlap data with GPU; run examples/benchmark_data.jl to see batches/sec you can supply."
    end
end

"""Load checkpoint; returns nothing if path does not exist. Errors propagate if file exists but is corrupt (no try-catch to avoid CUDA memory leaks)."""
function load_checkpoint(path::String)
    isfile(path) || return nothing
    jldopen(path, "r") do f
        step = f["step"]
        n_bins = f["n_bins"]
        model_state = f["model_state"]
        optimiser_state = get(f, "optimiser_state", nothing)  # optional; we reinit optimizer on resume
        dim = haskey(f, "dim") ? f["dim"] : nothing
        encoder_dim = haskey(f, "encoder_dim") ? f["encoder_dim"] : nothing
        encoder_layers = haskey(f, "encoder_layers") ? f["encoder_layers"] : (haskey(f, "n_layers") ? f["n_layers"] : nothing)
        n_heads = haskey(f, "n_heads") ? f["n_heads"] : nothing
        decoder_layers = haskey(f, "decoder_layers") ? f["decoder_layers"] : nothing
        cross_layers = haskey(f, "cross_layers") ? f["cross_layers"] : (haskey(f, "n_cross_layers") ? f["n_cross_layers"] : nothing)
        decoder_input_dropout = haskey(f, "decoder_input_dropout") ? Float32(f["decoder_input_dropout"]) : nothing
        self_attn_residual_scale = haskey(f, "self_attn_residual_scale") ? Float32(f["self_attn_residual_scale"]) : nothing
        ctc_weight = Float32(haskey(f, "ctc_weight") ? f["ctc_weight"] : 0.0)
        (; step, n_bins, model_state, optimiser_state, dim, encoder_dim, encoder_layers, n_heads, decoder_layers, cross_layers, decoder_input_dropout, self_attn_residual_scale, ctc_weight)
    end
end

function main()
    args = parse_commandline()
    rng = MersenneTwister(42)
    device = args.gpu ? gpu : cpu
    checkpoint_path = joinpath(args.checkpoint_dir, "checkpoint_latest.jld2")

    # Data via MorseSimulator: 200–900 Hz mel, ~10 Hz resolution, time resolution for 50 WPM. max_frames caps time for GPU memory/speed.
    cfg = SamplerConfig(; max_frames=args.max_frames)
    batch = generate_batch_fast(cfg, args.batch_size; rng)
    n_bins = size(batch.spectrogram, 1)

    loaded = load_checkpoint(checkpoint_path)
    step_start = 1
    decoder_layers = args.decoder_layers
    cross_layers = args.cross_layers
    decoder_input_dropout = args.decoder_input_dropout
    self_attn_residual_scale = args.self_attn_residual_scale
    if loaded !== nothing
        d = loaded
        dim = something(get(d, :dim, nothing), args.dim)
        encoder_dim_loaded = get(d, :encoder_dim, nothing)
        encoder_dim = encoder_dim_loaded !== nothing ? (encoder_dim_loaded == dim ? nothing : encoder_dim_loaded) : args.encoder_dim
        encoder_layers = something(get(d, :encoder_layers, nothing), args.encoder_layers)
        n_heads = something(get(d, :n_heads, nothing), args.n_heads)
        decoder_layers = something(get(d, :decoder_layers, nothing), args.decoder_layers)
        cross_layers = something(get(d, :cross_layers, nothing), args.cross_layers)
        decoder_input_dropout = Float32(something(get(d, :decoder_input_dropout, nothing), args.decoder_input_dropout))
        self_attn_residual_scale = Float32(something(get(d, :self_attn_residual_scale, nothing), 1.0f0))
        model = build_model(d.n_bins; dim, encoder_dim, n_heads, encoder_layers, decoder_layers, cross_layers, decoder_input_dropout, self_attn_residual_scale)
        # No try-catch: errors propagate (avoids CUDA memory leaks). Incompatible checkpoint => fix or remove file and rerun.
        Flux.loadmodel!(model, d.model_state)
        step_start = d.step + 1
        n_bins = d.n_bins
        # Fresh optimizer on resume (Muon state is not saved; adjust! applies LR schedule each step)
        opt = Flux.setup(Muon(eta=args.lr), model)
        if args.gpu
            model = device(model)
            opt = device(opt)
        end
        @info "Resumed from checkpoint" path=checkpoint_path from_step=d.step continuing_from=step_start dim=dim encoder_dim=encoder_dim encoder_layers=encoder_layers decoder_layers=decoder_layers cross_layers=cross_layers n_heads=n_heads self_attn_residual_scale=self_attn_residual_scale ctc_weight=args.ctc_weight
    else
        model = build_model(n_bins; dim=args.dim, encoder_dim=args.encoder_dim, n_heads=args.n_heads, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, cross_layers=args.cross_layers, decoder_input_dropout=args.decoder_input_dropout, self_attn_residual_scale=args.self_attn_residual_scale)
        if args.gpu
            model = device(model)
        end
        opt = Flux.setup(Muon(eta=args.lr), model)
    end

    # LR schedule: one-cycle (warmup then cosine decay) or constant. Warmup: fixed --warmup-steps or fraction (capped).
    warmup_steps = args.warmup_steps > 0 ? args.warmup_steps : (args.warmup_fraction > 0 ? min(ceil(Int, args.steps * args.warmup_fraction), 1000) : 0)
    lr_schedule = if warmup_steps > 0
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
    @info "Training" steps=args.steps device=(args.gpu ? "GPU" : "CPU") batch=args.batch_size max_frames=args.max_frames accum=args.accum_steps effective_batch=effective_batch dim=args.dim encoder_dim=args.encoder_dim encoder_layers=args.encoder_layers decoder_layers=args.decoder_layers cross_layers=args.cross_layers n_heads=args.n_heads decoder_input_dropout=args.decoder_input_dropout self_attn_residual_scale=args.self_attn_residual_scale n_params=n_params lr=args.lr warmup_steps=warmup_steps warmup_fraction=args.warmup_fraction save_every=args.save_every prefetch=args.prefetch prefetch_threads=Threads.nthreads() decode_every=args.decode_every ctc_weight=args.ctc_weight label_smoothing=args.label_smoothing

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

        ctc_kws = if args.ctc_weight > 0
            enc_len = div.(batch.input_lengths, MorseDecoder.ENCODER_DOWNSAMPLE)
            (; ctc_targets=prepare_ctc_targets(batch), input_lengths=enc_len, ctc_weight=args.ctc_weight)
        else
            (; ctc_weight=0.0f0)
        end

        result = Flux.withgradient(model) do m
            train_step(m, spec, decoder_input, decoder_target; ctc_kws..., label_smoothing=args.label_smoothing) / args.accum_steps
        end
        loss_sum += result.val * args.accum_steps

        if grads_accum === nothing
            grads_accum = result.grad[1]
        else
            grads_accum = Flux.fmap(accum_grad, grads_accum, result.grad[1])
        end
        accum_count += 1

        if accum_count == args.accum_steps
            eta = lr_schedule(min(step, args.steps))
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
                step_kw = (; step, loss=round(loss_avg; digits=4), lr=eta, steps_per_sec=round(steps_per_sec; digits=2))
                if steps_per_sec < 1.0
                    @info "step" step_kw... hint="Run with --benchmark 50 to see timing breakdown"
                else
                    @info "step" step_kw...
                end
            end
            if step % args.save_every == 0
                save_checkpoint(checkpoint_path, step, n_bins, model, opt; dim=args.dim, encoder_dim=args.encoder_dim, encoder_layers=args.encoder_layers, n_heads=args.n_heads, decoder_layers=args.decoder_layers, cross_layers=args.cross_layers, decoder_input_dropout=args.decoder_input_dropout, self_attn_residual_scale=args.self_attn_residual_scale, ctc_weight=args.ctc_weight)
            end
            if args.decode_every > 0 && step % args.decode_every == 0
                @info "decode" step=step n_samples=3
                run_decode_test(model, val_batches, device)
            end
        end
    end

    # Final update if we had a partial accumulation
    if accum_count > 0
        eta = lr_schedule(args.steps)
        adjust!(opt, eta)
        Flux.update!(opt, model, grads_accum)
    end

    # Save final checkpoint if we didn't already save at the last step (e.g. steps=60, save_every=30 → already saved at 60)
    if args.steps >= step_start && args.steps % args.save_every != 0
        save_checkpoint(checkpoint_path, args.steps, n_bins, model, opt; dim=args.dim, encoder_dim=args.encoder_dim, encoder_layers=args.encoder_layers, n_heads=args.n_heads, decoder_layers=args.decoder_layers, cross_layers=args.cross_layers, decoder_input_dropout=args.decoder_input_dropout, self_attn_residual_scale=args.self_attn_residual_scale, ctc_weight=args.ctc_weight)
    end

    # Final decode test on fixed validation (same as during training)
    @info "Decode test" n_spectrograms=3
    run_decode_test(model, val_batches, device)
end

main()
