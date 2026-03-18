#!/usr/bin/env julia
# Diagnostic script: run from repo root with same args as training.
#   julia --project=. examples/diagnose.jl --gpu [--ctc-weight 0.5 ...]
#
# Generates a few batches and prints diagnostics about:
# 1. CTC target feasibility (T >= 2*L+1)
# 2. Loss scale comparison (decoder CE vs raw CTC)
# 3. CTC predictions (blank dominance)
# 4. Encoder output statistics (temporal variation, cosine similarity, effective rank)
# 5. Gradient norms (encoder from CTC vs decoder)
# 6. Decoder behavior (cross-attention uniformity, teacher-forcing accuracy)
# 7. Input spectrogram statistics
# 8. Summary with actionable recommendations

if "--gpu" in ARGS
    using CUDA
    using cuDNN
end
using MorseDecoder
using MorseSimulator: DatasetConfig, DirectPath
using Flux
using CTCLoss
using Random
using Statistics
using LinearAlgebra
using JLD2

softmax(x) = (e = exp.(x .- maximum(x)); e ./ sum(e))

function parse_flag(args, flag, default, T=typeof(default))
    for (i, a) in enumerate(args)
        if a == flag && i < length(args)
            return parse(T, args[i+1])
        end
    end
    default
end

function parse_str_flag(args, flag, default)
    for (i, a) in enumerate(args)
        if a == flag && i < length(args)
            return args[i+1]
        end
    end
    default
end

function build_model(n_bins; dim=384, encoder_dim=nothing, n_heads=6,
                     encoder_layers=6, decoder_layers=4, cross_layers=6,
                     decoder_input_dropout=0.1f0, self_attn_residual_scale=1.0f0,
                     qk_norm=true)
    enc_dim = something(encoder_dim, dim)
    encoder = SpectrogramEncoder(n_bins, enc_dim, n_heads, encoder_layers; qk_norm)
    decoder = SpectrogramDecoder(MorseDecoder.VOCAB_SIZE, dim, n_heads, decoder_layers;
        n_cross_layers=cross_layers, decoder_input_dropout, self_attn_residual_scale, qk_norm)
    ctc_head = Flux.Dense(enc_dim => MorseDecoder.CTC_VOCAB_SIZE)
    encoder_proj = (enc_dim == dim) ? nothing : Flux.Dense(enc_dim => dim)
    SpectrogramEncoderDecoder(encoder, decoder, ctc_head, encoder_proj)
end

# ─── Fixed tree_norm: recursively collect all numeric array leaves ──────────

function collect_arrays!(arrays::Vector, x::AbstractArray{<:Number})
    push!(arrays, x)
    nothing
end
function collect_arrays!(arrays::Vector, x::NamedTuple)
    for v in values(x)
        collect_arrays!(arrays, v)
    end
    nothing
end
function collect_arrays!(arrays::Vector, x::Tuple)
    for v in x
        collect_arrays!(arrays, v)
    end
    nothing
end
function collect_arrays!(arrays::Vector, x::AbstractVector)
    # 1D numeric arrays (e.g. bias gradients on GPU) must be treated as leaves to avoid
    # scalar indexing when iterating; only recurse into non-array vectors (e.g. list of layers)
    if x isa AbstractArray{<:Number}
        push!(arrays, x)
    else
        for v in x
            collect_arrays!(arrays, v)
        end
    end
    nothing
end
collect_arrays!(::Vector, ::Nothing) = nothing
collect_arrays!(::Vector, ::Number) = nothing
collect_arrays!(::Vector, ::Any) = nothing

function tree_norm(tree)
    arrays = Any[]
    collect_arrays!(arrays, tree)
    isempty(arrays) && return 0.0
    # Move everything to CPU to avoid CUDA scalar indexing
    sqrt(sum(Float64(sum(abs2, Flux.cpu(a))) for a in arrays))
end

function main()
    gpu = "--gpu" in ARGS
    device = gpu ? Flux.gpu : Flux.cpu

    ctc_weight = parse_flag(ARGS, "--ctc-weight", 0.5f0, Float32)
    max_frames = parse_flag(ARGS, "--max-frames", 512, Int)
    batch_size = parse_flag(ARGS, "--batch", gpu ? 64 : 8, Int)
    checkpoint_dir = parse_str_flag(ARGS, "--checkpoint-dir", "checkpoints-lowLR-t-zeroCTC-test")
    checkpoint_path = joinpath(checkpoint_dir, "checkpoint_latest.jld2")

    rng = MersenneTwister(42)
    cfg = DatasetConfig(; path=DirectPath(), sample_rate=44100, fft_size=4096, hop_size=128,
        f_min=200.0, f_max=900.0, stations=2:4)

    # ── Generate batches first (determines n_bins) ───────────────────────────
    n_diag_batches = 5
    batches = [MorseDecoder.generate_training_batch(cfg, batch_size, max_frames; rng) for _ in 1:n_diag_batches]
    first_batch = batches[1]
    n_bins = size(first_batch.spectrogram, 1)

    println("="^70)
    println("MORSE DECODER TRAINING DIAGNOSTICS")
    println("="^70)
    println("  ctc_weight      = $ctc_weight")
    println("  max_frames      = $max_frames")
    println("  batch_size      = $batch_size")
    println("  n_bins (data)   = $n_bins")
    println("  device          = $(gpu ? "GPU" : "CPU")")
    println()

    # ── Load model ───────────────────────────────────────────────────────────
    model = nothing
    ckpt_step = nothing
    if isfile(checkpoint_path)
        println("Loading checkpoint: $checkpoint_path")
        d = jldopen(checkpoint_path, "r") do f
            step = f["step"]
            n_bins_ckpt = f["n_bins"]
            model_state = f["model_state"]
            dim = haskey(f, "dim") ? f["dim"] : 384
            encoder_dim_raw = haskey(f, "encoder_dim") ? f["encoder_dim"] : nothing
            encoder_dim = (encoder_dim_raw !== nothing && encoder_dim_raw != dim) ? encoder_dim_raw : nothing
            encoder_layers = haskey(f, "encoder_layers") ? f["encoder_layers"] : 6
            n_heads = haskey(f, "n_heads") ? f["n_heads"] : 6
            decoder_layers = haskey(f, "decoder_layers") ? f["decoder_layers"] : 4
            cross_layers = haskey(f, "cross_layers") ? f["cross_layers"] : 6
            decoder_input_dropout = haskey(f, "decoder_input_dropout") ? Float32(f["decoder_input_dropout"]) : 0.1f0
            self_attn_residual_scale = haskey(f, "self_attn_residual_scale") ? Float32(f["self_attn_residual_scale"]) : 1.0f0
            qk_norm = get(f, "qk_norm", true)
            (; step, n_bins=n_bins_ckpt, model_state, dim, encoder_dim, encoder_layers,
               n_heads, decoder_layers, cross_layers, decoder_input_dropout,
               self_attn_residual_scale, qk_norm)
        end
        if d.n_bins != n_bins
            error("Checkpoint n_bins=$(d.n_bins) != data n_bins=$n_bins. DatasetConfig mismatch.")
        end
        model = build_model(n_bins; dim=d.dim, encoder_dim=d.encoder_dim, n_heads=d.n_heads,
            encoder_layers=d.encoder_layers, decoder_layers=d.decoder_layers,
            cross_layers=d.cross_layers, decoder_input_dropout=d.decoder_input_dropout,
            self_attn_residual_scale=d.self_attn_residual_scale, qk_norm=d.qk_norm)
        Flux.loadmodel!(model, d.model_state)
        model = device(model)
        ckpt_step = d.step
        println("  Loaded from step $(d.step), dim=$(d.dim), enc_dim=$(something(d.encoder_dim, d.dim))")
    else
        println("No checkpoint found at $checkpoint_path — building fresh model")
        model = build_model(n_bins)
        model = device(model)
    end
    println()

    # ══════════════════════════════════════════════════════════════════════════
    # 1. CTC TARGET FEASIBILITY
    # ══════════════════════════════════════════════════════════════════════════
    println("="^70)
    println("1. CTC TARGET FEASIBILITY (T >= 2*L+1)")
    println("="^70)

    total_samples = 0
    infeasible_old = 0
    infeasible_new = 0
    label_lens_raw = Int[]
    label_lens_stripped = Int[]
    enc_lens_all = Int[]

    skip_old = Set((MorseDecoder.START_TOKEN_IDX, MorseDecoder.PAD_TOKEN_IDX, MorseDecoder.EOS_TOKEN_IDX, 0))
    skip_new = Set((MorseDecoder.START_TOKEN_IDX, MorseDecoder.PAD_TOKEN_IDX, MorseDecoder.EOS_TOKEN_IDX, 0,
                    MorseDecoder.SPEAKER_1_IDX, MorseDecoder.SPEAKER_2_IDX, MorseDecoder.SPEAKER_3_IDX,
                    MorseDecoder.SPEAKER_4_IDX, MorseDecoder.SPEAKER_5_IDX, MorseDecoder.SPEAKER_6_IDX))

    for batch in batches
        B = size(batch.targets, 1)
        enc_lengths = div.(batch.input_lengths, MorseDecoder.ENCODER_DOWNSAMPLE)
        for b in 1:B
            total_samples += 1
            pfx = batch.prefix_lengths[b]
            tgt_end = batch.target_lengths[b]
            chunk_tgt = @view batch.targets[b, pfx+1:tgt_end]

            raw_old = [t for t in chunk_tgt if t ∉ skip_old]
            raw_new = [t for t in chunk_tgt if t ∉ skip_new]
            T_b = enc_lengths[b]
            L_max = max(0, div(T_b - 1, 2))

            push!(label_lens_raw, length(raw_old))
            push!(label_lens_stripped, length(raw_new))
            push!(enc_lens_all, T_b)

            if length(raw_old) > L_max
                infeasible_old += 1
            end
            if length(raw_new) > L_max
                infeasible_new += 1
            end
        end
    end

    l_max_all = max.(0, div.(enc_lens_all .- 1, 2))
    println("  Total samples analyzed: $total_samples")
    println()
    println("  Encoder frames (T):       min=$(minimum(enc_lens_all))  median=$(round(median(enc_lens_all); digits=1))  max=$(maximum(enc_lens_all))")
    println("  L_max = (T-1)/2:          min=$(minimum(l_max_all))  median=$(round(median(l_max_all); digits=1))  max=$(maximum(l_max_all))")
    println("  Label len (with speakers): min=$(minimum(label_lens_raw))  median=$(round(median(label_lens_raw); digits=1))  max=$(maximum(label_lens_raw))")
    println("  Label len (no speakers):   min=$(minimum(label_lens_stripped))  median=$(round(median(label_lens_stripped); digits=1))  max=$(maximum(label_lens_stripped))")
    println()
    println("  Infeasible (with speakers, current code truncates these): $(infeasible_old)/$(total_samples) = $(round(100*infeasible_old/total_samples; digits=1))%")
    println("  Infeasible (no speakers, after fix):                      $(infeasible_new)/$(total_samples) = $(round(100*infeasible_new/total_samples; digits=1))%")
    println()

    println("  Sample breakdown (first batch, first 8):")
    batch = first_batch
    enc_lengths = div.(batch.input_lengths, MorseDecoder.ENCODER_DOWNSAMPLE)
    for b in 1:min(8, size(batch.targets, 1))
        pfx = batch.prefix_lengths[b]
        tgt_end = batch.target_lengths[b]
        chunk_tgt = @view batch.targets[b, pfx+1:tgt_end]
        raw_old = [t for t in chunk_tgt if t ∉ skip_old]
        raw_new = [t for t in chunk_tgt if t ∉ skip_new]
        T_b = enc_lengths[b]
        L_max = max(0, div(T_b - 1, 2))
        n_speaker = length(raw_old) - length(raw_new)
        status = length(raw_old) > L_max ? "TRUNCATED $(length(raw_old))→$(min(length(raw_old), L_max))" : "ok"
        status_new = length(raw_new) > L_max ? " (still infeasible after strip)" : ""
        println("    b=$b: T=$T_b  L_max=$L_max  labels=$(length(raw_old))($(n_speaker) speakers)  stripped=$(length(raw_new))  [$status$status_new]")
    end
    println()

    # ══════════════════════════════════════════════════════════════════════════
    # 2. LOSS SCALE COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    println("="^70)
    println("2. LOSS SCALE COMPARISON")
    println("="^70)

    batch = first_batch
    spec, decoder_input, decoder_target, loss_mask = MorseDecoder.prepare_training_batch(batch)
    spec_d = device(spec)
    decoder_input_d = device(decoder_input)
    decoder_target_d = device(decoder_target)
    loss_mask_d = device(loss_mask)

    enc_mem, dec_mem = MorseDecoder.encode(model, spec_d)
    logits = model.decoder(decoder_input_d, dec_mem)
    decoder_loss = MorseDecoder.sequence_cross_entropy(logits, decoder_target_d, loss_mask_d)
    decoder_loss_val = Float64(Flux.cpu(decoder_loss))

    enc_lengths = div.(batch.input_lengths, MorseDecoder.ENCODER_DOWNSAMPLE)
    ctc_targets = MorseDecoder.prepare_ctc_targets(batch, enc_lengths)
    ctc_logits = model.ctc_head(enc_mem)

    ctc_raw_val = Float64(Flux.cpu(CTCLoss.ctc_loss_batched(ctc_logits, ctc_targets, enc_lengths, MorseDecoder.CTC_BLANK_IDX)))

    avg_frames = sum(enc_lengths) / length(enc_lengths)
    ctc_per_frame = ctc_raw_val / avg_frames

    println("  Decoder CE (per-token):        $(round(decoder_loss_val; digits=4))")
    println("  CTC loss (per-sample, raw):    $(round(ctc_raw_val; digits=4))")
    println("  CTC loss (per-frame):          $(round(ctc_per_frame; digits=4))")
    println("  Avg encoder frames:            $(round(avg_frames; digits=1))")
    println()
    println("  Ratio CTC_raw / decoder_CE:    $(round(ctc_raw_val / max(decoder_loss_val, 1e-8); digits=1))x")
    println("  Ratio CTC_perframe / decoder:  $(round(ctc_per_frame / max(decoder_loss_val, 1e-8); digits=1))x")
    println()
    println("  With ctc_weight=$ctc_weight:")
    println("    Current code:  total = $(round(decoder_loss_val + ctc_weight * ctc_raw_val; digits=4))  (decoder=$(round(decoder_loss_val; digits=4)) + ctc=$(round(ctc_weight * ctc_raw_val; digits=4)))")
    println("    Per-frame fix: total = $(round(decoder_loss_val + ctc_weight * ctc_per_frame; digits=4))  (decoder=$(round(decoder_loss_val; digits=4)) + ctc=$(round(ctc_weight * ctc_per_frame; digits=4)))")
    println()

    # ══════════════════════════════════════════════════════════════════════════
    # 3. CTC PREDICTIONS (blank dominance)
    # ══════════════════════════════════════════════════════════════════════════
    println("="^70)
    println("3. CTC PREDICTIONS (blank dominance check)")
    println("="^70)

    ctc_logits_cpu = Flux.cpu(ctc_logits)
    blank_idx = MorseDecoder.CTC_BLANK_IDX
    total_frames_checked = 0
    blank_argmax_count = 0
    blank_prob_sum = 0.0

    for b in 1:min(size(ctc_logits_cpu, 3), batch_size)
        T_b = enc_lengths[b]
        for t in 1:T_b
            total_frames_checked += 1
            frame_logits = ctc_logits_cpu[:, t, b]
            probs = softmax(frame_logits)
            blank_prob_sum += probs[blank_idx]
            if argmax(frame_logits) == blank_idx
                blank_argmax_count += 1
            end
        end
    end

    println("  Frames where blank is argmax: $(blank_argmax_count)/$(total_frames_checked) = $(round(100*blank_argmax_count/total_frames_checked; digits=1))%")
    println("  Mean blank probability:       $(round(blank_prob_sum/total_frames_checked; digits=4))")
    println()

    println("  CTC greedy decode (first 3 samples):")
    for b in 1:min(3, size(spec, 2))
        s = spec_d[:, b:b, :]
        seqs = MorseDecoder.ctc_greedy_decode(model, s; input_lengths=[batch.input_lengths[b]])
        decoded = MorseDecoder.token_ids_to_label(seqs[1])
        pfx = batch.prefix_lengths[b]
        truth_ids = batch.targets[b, pfx+1:batch.target_lengths[b]]
        truth = MorseDecoder.token_ids_to_label(truth_ids)
        println("    b=$b truth: $(isempty(truth) ? "(empty)" : truth)")
        println("        ctc:   $(isempty(decoded) ? "(empty/all-blank)" : decoded)")
    end
    println()

    # ══════════════════════════════════════════════════════════════════════════
    # 4. ENCODER OUTPUT STATISTICS
    # ══════════════════════════════════════════════════════════════════════════
    println("="^70)
    println("4. ENCODER OUTPUT STATISTICS")
    println("="^70)

    enc_cpu = Float64.(Flux.cpu(enc_mem))
    println("  Shape: $(size(enc_cpu))  (dim, time, batch)")
    println("  Mean:     $(round(mean(enc_cpu); sigdigits=4))")
    println("  Std:      $(round(std(enc_cpu); sigdigits=4))")
    println("  Min:      $(round(minimum(enc_cpu); sigdigits=4))")
    println("  Max:      $(round(maximum(enc_cpu); sigdigits=4))")
    println("  Abs mean: $(round(mean(abs, enc_cpu); sigdigits=4))")
    println()

    dim_enc, T_enc, B = size(enc_cpu)
    temporal_stds = Float64[]
    for b in 1:min(B, 10)
        T_b = enc_lengths[b]
        if T_b > 1
            for d in 1:dim_enc
                push!(temporal_stds, std(@view enc_cpu[d, 1:T_b, b]))
            end
        end
    end
    println("  Temporal variation (std across time per dim):")
    println("    Mean: $(round(mean(temporal_stds); sigdigits=4))")
    println("    Min:  $(round(minimum(temporal_stds); sigdigits=4))")
    println("    Max:  $(round(maximum(temporal_stds); sigdigits=4))")

    cos_sims = Float64[]
    for b in 1:min(B, 10)
        T_b = enc_lengths[b]
        for t in 1:T_b-1
            a = @view enc_cpu[:, t, b]
            c = @view enc_cpu[:, t+1, b]
            dot_val = sum(a .* c)
            na = sqrt(sum(a .^ 2))
            nc = sqrt(sum(c .^ 2))
            if na > 1e-8 && nc > 1e-8
                push!(cos_sims, dot_val / (na * nc))
            end
        end
    end
    if !isempty(cos_sims)
        println("  Adjacent frame cosine similarity:")
        println("    Mean: $(round(mean(cos_sims); sigdigits=4))  (1.0 = all frames identical = BAD)")
        println("    Min:  $(round(minimum(cos_sims); sigdigits=4))")
        println("    Max:  $(round(maximum(cos_sims); sigdigits=4))")
    end
    println()

    println("  Encoder effective rank (SVD on first 5 samples):")
    for b in 1:min(5, B)
        T_b = enc_lengths[b]
        T_b < 2 && continue
        mat = enc_cpu[:, 1:T_b, b]
        s = svdvals(mat)
        s_norm = s ./ sum(s)
        s_norm = s_norm[s_norm .> 1e-12]
        eff_rank = exp(-sum(p * log(p) for p in s_norm))
        top1_pct = 100.0 * s[1]^2 / sum(s .^ 2)
        top5_pct = 100.0 * sum(s[1:min(5, length(s))] .^ 2) / sum(s .^ 2)
        println("    b=$b: eff_rank=$(round(eff_rank; digits=1))/$(min(dim_enc, T_b))  top-1 SV=$(round(top1_pct; digits=1))%  top-5 SV=$(round(top5_pct; digits=1))% of variance")
    end
    println()

    stride64_mean = NaN
    println("  Cosine similarity at various strides (first 5 samples):")
    for stride in [1, 4, 16, 64]
        sims = Float64[]
        for b in 1:min(5, B)
            T_b = enc_lengths[b]
            for t in 1:T_b-stride
                a = @view enc_cpu[:, t, b]
                c = @view enc_cpu[:, t+stride, b]
                dot_val = sum(a .* c)
                na = sqrt(sum(a .^ 2))
                nc = sqrt(sum(c .^ 2))
                if na > 1e-8 && nc > 1e-8
                    push!(sims, dot_val / (na * nc))
                end
            end
        end
        if !isempty(sims)
            m = mean(sims)
            if stride == 64
                stride64_mean = m
            end
            println("    stride=$stride:  mean=$(round(m; sigdigits=4))  min=$(round(minimum(sims); sigdigits=4))  max=$(round(maximum(sims); sigdigits=4))")
        end
    end
    println()

    # ══════════════════════════════════════════════════════════════════════════
    # 5. GRADIENT NORMS
    # ══════════════════════════════════════════════════════════════════════════
    println("="^70)
    println("5. GRADIENT NORMS")
    println("="^70)

    # Initialize before try so summary section always has values
    dec_encoder_norm = NaN
    dec_decoder_norm = NaN
    ctc_encoder_norm = NaN
    ctc_head_norm = NaN

    try
        dec_grad = Flux.gradient(model) do m
            enc_m, dec_m = MorseDecoder.encode(m, spec_d)
            lg = m.decoder(decoder_input_d, dec_m)
            MorseDecoder.sequence_cross_entropy(lg, decoder_target_d, loss_mask_d)
        end[1]

        dec_encoder_norm = tree_norm(dec_grad.encoder)
        dec_decoder_norm = tree_norm(dec_grad.decoder)

        ctc_grad = Flux.gradient(model) do m
            enc_m, _ = MorseDecoder.encode(m, spec_d)
            lg = m.ctc_head(enc_m)
            CTCLoss.ctc_loss_batched(lg, ctc_targets, enc_lengths, MorseDecoder.CTC_BLANK_IDX)
        end[1]

        ctc_encoder_norm = tree_norm(ctc_grad.encoder)
        ctc_head_norm = tree_norm(ctc_grad.ctc_head)

        println("  From decoder CE loss only:")
        println("    encoder grad norm: $(round(dec_encoder_norm; sigdigits=4))")
        println("    decoder grad norm: $(round(dec_decoder_norm; sigdigits=4))")
        println()
        println("  From CTC loss only (raw, no weight):")
        println("    encoder grad norm: $(round(ctc_encoder_norm; sigdigits=4))")
        println("    ctc_head grad norm: $(round(ctc_head_norm; sigdigits=4))")
        println()
        println("  Ratio CTC_encoder / decoder_encoder:  $(round(ctc_encoder_norm / max(dec_encoder_norm, 1e-12); digits=1))x")
        println("  With ctc_weight=$ctc_weight (current):  $(round(ctc_weight * ctc_encoder_norm / max(dec_encoder_norm, 1e-12); digits=1))x")
        println("  With ctc_weight=$ctc_weight (per-frame): $(round(ctc_weight * ctc_encoder_norm / avg_frames / max(dec_encoder_norm, 1e-12); digits=1))x")

        if dec_encoder_norm < 1e-6
            println()
            println("  🔴 Decoder→encoder gradient is near zero!")
            println("     Cross-attention is not propagating gradients to the encoder.")
        end
    catch e
        println("  ERROR computing gradient norms: $e")
    end
    println()

    # ══════════════════════════════════════════════════════════════════════════
    # 6. DECODER BEHAVIOR ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    println("="^70)
    println("6. DECODER BEHAVIOR ANALYSIS")
    println("="^70)

    # 6a. Teacher-forcing accuracy
    logits_cpu = Flux.cpu(logits)
    target_cpu = Flux.cpu(decoder_target_d)
    mask_cpu = Flux.cpu(loss_mask_d)

    correct = 0
    total_masked = 0
    for b in 1:size(target_cpu, 2)
        for t in 1:size(target_cpu, 1)
            mask_cpu[t, b] > 0.5f0 || continue
            total_masked += 1
            pred = argmax(logits_cpu[:, t, b])
            if pred == target_cpu[t, b]
                correct += 1
            end
        end
    end
    teacher_acc = total_masked > 0 ? correct / total_masked : 0.0
    println("  Teacher-forcing accuracy: $(round(100*teacher_acc; digits=1))% ($correct/$total_masked masked positions)")
    println()

    # 6b. Repetition analysis in greedy decode
    println("  Greedy decode repetition analysis (first 3 samples):")
    for b in 1:min(3, size(spec, 2))
        s = spec_d[:, b:b, :]
        ids = MorseDecoder.decode_autoregressive(model, s; max_len=128, to_device=device)
        ids_cpu = Flux.cpu(ids)
        seq = [ids_cpu[i, 1] for i in 1:size(ids_cpu, 1)]
        content = [id for id in seq if id != MorseDecoder.START_TOKEN_IDX &&
                                       id != MorseDecoder.PAD_TOKEN_IDX &&
                                       id != MorseDecoder.EOS_TOKEN_IDX && id != 0]
        if length(content) > 2
            unique_tokens = length(unique(content))
            total_tokens = length(content)
            max_repeat_len = 0
            for sublen in 1:min(20, total_tokens ÷ 2)
                sub = content[1:sublen]
                repeats = 0
                for start in 1:sublen:total_tokens-sublen+1
                    if content[start:start+sublen-1] == sub
                        repeats += 1
                    else
                        break
                    end
                end
                if repeats > 1
                    max_repeat_len = max(max_repeat_len, repeats)
                end
            end
            decoded = MorseDecoder.token_ids_to_label(content)
            show_str = length(decoded) > 80 ? decoded[1:80] * "..." : decoded
            println("    b=$b: $(total_tokens) tokens, $(unique_tokens) unique, longest prefix repeat=$(max_repeat_len)x")
            println("         $(show_str)")
        else
            println("    b=$b: $(length(content)) tokens (too short for analysis)")
        end
    end
    println()

    # 6c. Encoder-independence test
    println("  Encoder-independence test (decoder with zeroed encoder memory):")
    try
        for b in 1:min(2, size(spec, 2))
            s = spec_d[:, b:b, :]
            ids_normal = MorseDecoder.decode_autoregressive(model, s; max_len=64, to_device=device)
            normal_seq = [Flux.cpu(ids_normal)[i, 1] for i in 1:size(ids_normal, 1)]

            _, mem_real = MorseDecoder.encode(model, s)
            zero_mem = device(zeros(Float32, size(Flux.cpu(mem_real))))
            ids_buf = device(fill(MorseDecoder.START_TOKEN_IDX, 64, 1))
            ids_zero = MorseDecoder.autoregressive_loop(model, zero_mem, ids_buf, 1; max_len=64, to_device=device)
            zero_seq = [Flux.cpu(ids_zero)[i, 1] for i in 1:size(ids_zero, 1)]

            min_len = min(length(normal_seq), length(zero_seq))
            matching = sum(normal_seq[i] == zero_seq[i] for i in 1:min_len)
            pct_match = round(100 * matching / max(min_len, 1); digits=1)
            normal_str = MorseDecoder.token_ids_to_label(normal_seq)
            zero_str = MorseDecoder.token_ids_to_label(zero_seq)
            show_n = min(60, length(normal_str))
            show_z = min(60, length(zero_str))
            println("    b=$b: $(pct_match)% token overlap ($(matching)/$(min_len))")
            println("         normal: $(normal_str[1:show_n])$(length(normal_str) > 60 ? "..." : "")")
            println("         zeroed: $(zero_str[1:show_z])$(length(zero_str) > 60 ? "..." : "")")
        end
    catch e
        println("  ERROR in encoder-independence test: $e")
    end
    println()

    # 6d. Random encoder memory test
    println("  Random encoder memory test (is decoder sensitive to encoder at all?):")
    try
        s = spec_d[:, 1:1, :]
        ids_normal = MorseDecoder.decode_autoregressive(model, s; max_len=64, to_device=device)
        normal_seq = [Flux.cpu(ids_normal)[i, 1] for i in 1:size(ids_normal, 1)]

        _, mem_real = MorseDecoder.encode(model, s)
        mem_cpu = Flux.cpu(mem_real)
        random_mem = device(Float32.(randn(size(mem_cpu)) .* std(mem_cpu) .+ mean(mem_cpu)))
        ids_buf = device(fill(MorseDecoder.START_TOKEN_IDX, 64, 1))
        ids_rand = MorseDecoder.autoregressive_loop(model, random_mem, ids_buf, 1; max_len=64, to_device=device)
        rand_seq = [Flux.cpu(ids_rand)[i, 1] for i in 1:size(ids_rand, 1)]

        min_len = min(length(normal_seq), length(rand_seq))
        matching = sum(normal_seq[i] == rand_seq[i] for i in 1:min_len)
        pct_match = round(100 * matching / max(min_len, 1); digits=1)
        normal_str = MorseDecoder.token_ids_to_label(normal_seq)
        rand_str = MorseDecoder.token_ids_to_label(rand_seq)
        show_n = min(60, length(normal_str))
        show_r = min(60, length(rand_str))
        println("    $(pct_match)% token overlap with random memory ($(matching)/$(min_len))")
        println("    normal: $(normal_str[1:show_n])$(length(normal_str) > 60 ? "..." : "")")
        println("    random: $(rand_str[1:show_r])$(length(rand_str) > 60 ? "..." : "")")
        if pct_match > 80
            println("    ⚠  Decoder ignoring encoder entirely (language model collapse).")
        elseif pct_match > 50
            println("    ⚠  Weak encoder dependence.")
        else
            println("    ✓  Decoder output differs substantially — encoder IS being used.")
        end
    catch e
        println("  ERROR in random memory test: $e")
    end
    println()

    # 6e. Quality comparison
    println("  Output quality comparison:")
    try
        s = spec_d[:, 1:1, :]
        ids_normal = MorseDecoder.decode_autoregressive(model, s; max_len=64, to_device=device)
        normal_seq = [Flux.cpu(ids_normal)[i, 1] for i in 1:size(ids_normal, 1)]
        normal_content = [id for id in normal_seq if id != MorseDecoder.START_TOKEN_IDX &&
                                                      id != MorseDecoder.PAD_TOKEN_IDX &&
                                                      id != MorseDecoder.EOS_TOKEN_IDX && id != 0]

        _, mem_real = MorseDecoder.encode(model, s)
        zero_mem = device(zeros(Float32, size(Flux.cpu(mem_real))))
        ids_buf = device(fill(MorseDecoder.START_TOKEN_IDX, 64, 1))
        ids_zero = MorseDecoder.autoregressive_loop(model, zero_mem, ids_buf, 1; max_len=64, to_device=device)
        zero_seq = [Flux.cpu(ids_zero)[i, 1] for i in 1:size(ids_zero, 1)]
        zero_content = [id for id in zero_seq if id != MorseDecoder.START_TOKEN_IDX &&
                                                  id != MorseDecoder.PAD_TOKEN_IDX &&
                                                  id != MorseDecoder.EOS_TOKEN_IDX && id != 0]

        normal_diversity = length(normal_content) > 0 ? length(unique(normal_content)) / length(normal_content) : 0.0
        zero_diversity = length(zero_content) > 0 ? length(unique(zero_content)) / length(zero_content) : 0.0

        normal_has_ts = MorseDecoder.TS_TOKEN_IDX in normal_content
        normal_has_te = MorseDecoder.TE_TOKEN_IDX in normal_content
        zero_has_ts = MorseDecoder.TS_TOKEN_IDX in zero_content
        zero_has_te = MorseDecoder.TE_TOKEN_IDX in zero_content

        println("    Normal encoder:  $(length(normal_content)) tokens, $(length(unique(normal_content))) unique ($(round(100*normal_diversity; digits=0))% diverse), TS=$(normal_has_ts), TE=$(normal_has_te)")
        println("    Zeroed encoder:  $(length(zero_content)) tokens, $(length(unique(zero_content))) unique ($(round(100*zero_diversity; digits=0))% diverse), TS=$(zero_has_ts), TE=$(zero_has_te)")

        if zero_diversity > normal_diversity * 2 && zero_has_ts && zero_has_te
            println("    🔴 Zeroed encoder produces BETTER output (more diverse, has structure)!")
            println("       The real encoder is actively harming decoder generation.")
        elseif zero_diversity > normal_diversity
            println("    ⚠  Zeroed encoder produces more diverse output than real encoder.")
        else
            println("    ✓  Real encoder output produces better or equal output.")
        end
    catch e
        println("  ERROR in quality comparison: $e")
    end
    println()

    # ══════════════════════════════════════════════════════════════════════════
    # 7. INPUT SPECTROGRAM STATISTICS
    # ══════════════════════════════════════════════════════════════════════════
    println("="^70)
    println("7. INPUT SPECTROGRAM STATISTICS")
    println("="^70)

    spec_cpu = Flux.cpu(spec_d)
    println("  Shape: $(size(spec_cpu))  (n_bins, batch, time)")
    println("  Mean:  $(round(mean(spec_cpu); sigdigits=4))")
    println("  Std:   $(round(std(spec_cpu); sigdigits=4))")
    println("  Min:   $(round(minimum(spec_cpu); sigdigits=4))")
    println("  Max:   $(round(maximum(spec_cpu); sigdigits=4))")

    spec_time_std = Float64[]
    for b in 1:min(5, size(spec_cpu, 2))
        T_b = batch.input_lengths[b]
        for f in 1:size(spec_cpu, 1)
            push!(spec_time_std, std(@view spec_cpu[f, b, 1:T_b]))
        end
    end
    println("  Temporal variation (std per freq bin): mean=$(round(mean(spec_time_std); sigdigits=4))  min=$(round(minimum(spec_time_std); sigdigits=4))  max=$(round(maximum(spec_time_std); sigdigits=4))")

    active_bins = 0
    for f in 1:size(spec_cpu, 1)
        if maximum(spec_cpu[f, 1, :]) > -5.0
            active_bins += 1
        end
    end
    println("  Active freq bins (max > -5.0 in log10): $active_bins / $(size(spec_cpu, 1))")
    println()

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY & RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════════════════
    println("="^70)
    println("SUMMARY & RECOMMENDATIONS")
    println("="^70)
    issues = String[]

    if ctc_weight * ctc_raw_val > 5 * decoder_loss_val
        push!(issues, "⚠  CTC raw loss $(round(ctc_weight * ctc_raw_val / max(decoder_loss_val, 1e-8); digits=1))x decoder (per-frame normalization in training.jl handles this)")
    end

    blank_ratio = blank_argmax_count / total_frames_checked
    if blank_ratio > 0.95
        push!(issues, "⚠  CTC blank $(round(100*blank_ratio; digits=1))% — fully collapsed")
    elseif blank_ratio > 0.9
        push!(issues, "⚠  CTC blank $(round(100*blank_ratio; digits=1))% — mostly blank (expected: median 3 chars / 256 frames)")
    end

    mean_cos = !isempty(cos_sims) ? mean(cos_sims) : 0.0
    if mean_cos > 0.99
        push!(issues, "🔴 Encoder fully collapsed: adjacent cosine sim $(round(mean_cos; sigdigits=4))")
    elseif mean_cos > 0.95
        push!(issues, "⚠  High adjacent cosine sim $(round(mean_cos; sigdigits=4)) (check stride analysis for true severity)")
    end

    if infeasible_old / total_samples > 0.1
        push!(issues, "⚠  $(round(100*infeasible_old/total_samples; digits=1))% CTC targets truncated")
    end

    if !isnan(dec_encoder_norm) && !isnan(ctc_encoder_norm)
        if dec_encoder_norm < 1e-6
            push!(issues, "🔴 Decoder→encoder gradient ≈ 0 — cross-attention not training encoder")
        end
        if ctc_weight * ctc_encoder_norm > 10 * dec_encoder_norm && dec_encoder_norm > 1e-6
            push!(issues, "⚠  CTC gradient $(round(ctc_weight * ctc_encoder_norm / max(dec_encoder_norm, 1e-12); digits=1))x decoder on encoder")
        end
    else
        push!(issues, "⚠  Gradient norms failed to compute (see section 5)")
    end

    if teacher_acc > 0.90
        push!(issues, "⚠  Teacher-forcing $(round(100*teacher_acc; digits=1))% on only $(total_masked) positions (short seqs make this misleadingly high)")
    end

    if isempty(issues)
        println("  ✓  No obvious issues detected")
    else
        for issue in issues
            println("  $issue")
        end
    end
    println()

    println("  INTERPRETATION:")
    println("  ───────────────")
    println()
    println("  Cross-attention uses Independent RoPE (Q=decoder positions, K=encoder positions).")
    println()
    # Data-driven encoder health (use actual mean_cos and stride-64)
    encoder_collapsed = !isempty(cos_sims) && mean(cos_sims) > 0.99 && (isnan(stride64_mean) || stride64_mean > 0.95)
    if encoder_collapsed
        println("  Encoder health: 🔴 FULLY COLLAPSED")
        println("    - Adjacent cosine sim $(round(mean(cos_sims); sigdigits=4)), stride-64 still $(round(stride64_mean; sigdigits=4))")
        println("    - Encoder output is almost the same at every timestep → no useful temporal signal.")
        println("    - CTC 100% blank and decoder repetition are consequences. Likely cause: CTC")
        println("      gradient dominating (see ratio above) and pushing encoder to a constant.")
    elseif !isempty(cos_sims) && mean(cos_sims) > 0.95
        s64 = isnan(stride64_mean) ? "?" : string(round(stride64_mean; sigdigits=4))
        println("  Encoder health: PARTIALLY OK")
        println("    - Adjacent cosine sim $(round(mean(cos_sims); sigdigits=4)); stride-64 mean = $s64")
        println("    - If stride-64 drops well below adjacent, encoder has some temporal structure.")
        println("    - High adjacent sim is expected: CW changes slowly vs frame rate (344 Hz).")
    else
        println("  Encoder health: OK (temporal variation present)")
    end
    println()
    println("  Decoder–encoder coupling: see section 6 (zeroed/random memory tests).")
    println("  If zeroed encoder looks better: cross-attention learned degenerate patterns;")
    println("  with RoPE on K, persistence suggests training dynamics or unrecoverable checkpoint.")
    println()

    println("  RECOMMENDED FIXES (priority order):")
    println("  ────────────────────────────────────")
    println()
    step_known = ckpt_step !== nothing
    if encoder_collapsed
        println("  1. REDUCE --ctc-weight (e.g. 0.05–0.1) so decoder gradient can train encoder.")
        println("     (Training uses per-frame CTC; if ratio in section 2 is still high, lower weight.)")
        println()
        if step_known && ckpt_step < 10000
            println("  2. TRAIN LONGER: this checkpoint is only $(ckpt_step) steps; encoder may recover.")
            println()
            println("  3. LOWER LR / LONGER WARMUP if starting a new run.")
        else
            println("  2. LOWER LR: --lr 1e-4 and LONGER WARMUP: --warmup-steps 2000")
            println()
            println("  3. TRAIN FROM SCRATCH with balanced CTC weight.")
        end
    else
        println("  1. LOWER LR: --lr 1e-4 (vs 1e-3) if cross-attention is unstable.")
        println()
        println("  2. LONGER WARMUP: --warmup-steps 2000")
        println()
        println("  3. INCREASE DECODER INPUT DROPOUT: --decoder-input-dropout 0.3")
        println()
        println("  4. If collapse persists: train from scratch or add extra position signal on encoder.")
    end
    println()
end

main()
