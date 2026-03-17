#!/usr/bin/env julia
# Diagnose encoder input quality and CTC target feasibility.
# Run from repo root:
#   julia --project=. examples/diagnose-encoder-decoder.jl [--checkpoint path/to/checkpoint.jld2] [--gpu]

using MorseDecoder
using MorseSimulator: DatasetConfig, DirectPath
using Random
using Statistics
using LinearAlgebra: dot, norm
using CairoMakie

# Optional: GPU + checkpoint
use_gpu = "--gpu" in ARGS
checkpoint_path = let idx = findfirst(==("--checkpoint"), ARGS)
    idx !== nothing && idx < length(ARGS) ? ARGS[idx+1] : nothing
end

if use_gpu
    using CUDA
    using cuDNN
    using Flux
    device = gpu
else
    using Flux
    device = cpu
end
using Flux: softmax

# Same config as training (linear band 200–900 Hz, ~66 bins)
cfg = DatasetConfig(; path = DirectPath(), sample_rate = 44100, fft_size = 4096, hop_size = 128,
    f_min = 200.0, f_max = 900.0, stations = 2:4)

rng = MersenneTwister(42)

println("=" ^ 70)
println("PART 1: Spectrogram statistics (10 samples)")
println("=" ^ 70)

for i in 1:10
    sample = generate_sample(cfg; rng)
    spec = sample.spectrogram  # (n_bins, n_frames) Float32
    n_bins, n_frames = size(spec)
    label = token_ids_to_label(sample.token_ids)
    n_tokens = length(sample.token_ids)

    # Per-bin statistics
    bin_means = vec(mean(spec; dims=2))
    bin_stds = vec(std(spec; dims=2))

    println("\n--- Sample $i ---")
    println("  spec size:    ($n_bins, $n_frames)")
    println("  token count:  $n_tokens")
    println("  label:        $(first(label, 80))...")
    println("  spec min:     $(minimum(spec))")
    println("  spec max:     $(maximum(spec))")
    println("  spec mean:    $(mean(spec))")
    println("  spec std:     $(std(spec))")
    println("  spec % zeros: $(round(100 * count(==(0f0), spec) / length(spec); digits=1))%")
    println("  spec % < -5:  $(round(100 * count(<(-5f0), spec) / length(spec); digits=1))%")
    println("  bin mean range: [$(round(minimum(bin_means); digits=3)), $(round(maximum(bin_means); digits=3))]")
    println("  bin std range:  [$(round(minimum(bin_stds); digits=3)), $(round(maximum(bin_stds); digits=3))]")

    # How much variance is there across time? (if all frames look the same, encoder can't learn)
    frame_means = vec(mean(spec; dims=1))
    println("  frame mean range: [$(round(minimum(frame_means); digits=3)), $(round(maximum(frame_means); digits=3))]")
    println("  frame mean std:   $(round(std(frame_means); digits=4))")
end

println("\n" * "=" ^ 70)
println("PART 2: Training batch CTC target analysis (20 batches)")
println("=" ^ 70)

let
    total_empty_ctc = 0
    total_truncated_ctc = 0
    total_samples = 0
    all_ctc_lengths = Int[]
    all_enc_lengths = Int[]
    all_raw_lengths = Int[]

    for b in 1:20
        batch = generate_training_batch(cfg, 64, 512; rng)
        enc_lengths = div.(batch.input_lengths, MorseDecoder.ENCODER_DOWNSAMPLE)

        B = size(batch.targets, 1)
        skip = Set((START_TOKEN_IDX, PAD_TOKEN_IDX, EOS_TOKEN_IDX, 0))

        for i in 1:B
            pfx = batch.prefix_lengths[i]
            tgt_end = batch.target_lengths[i]
            chunk_tgt = @view batch.targets[i, pfx+1:tgt_end]
            raw = [t for t in chunk_tgt if t ∉ skip]
            T_b = enc_lengths[i]
            L_max = max(0, div(T_b - 1, 2))

            push!(all_raw_lengths, length(raw))
            push!(all_enc_lengths, T_b)

            if L_max == 0 || isempty(raw)
                total_empty_ctc += 1
                push!(all_ctc_lengths, 0)
            elseif length(raw) > L_max
                total_truncated_ctc += 1
                push!(all_ctc_lengths, L_max)
            else
                push!(all_ctc_lengths, length(raw))
            end
            total_samples += 1
        end
    end

    println("\n  Total samples:           $total_samples")
    println("  Empty CTC targets:       $total_empty_ctc ($(round(100 * total_empty_ctc / total_samples; digits=1))%)")
    println("  Truncated CTC targets:   $total_truncated_ctc ($(round(100 * total_truncated_ctc / total_samples; digits=1))%)")
    println("  Encoder frame lengths:   min=$(minimum(all_enc_lengths)), max=$(maximum(all_enc_lengths)), mean=$(round(mean(all_enc_lengths); digits=1))")
    println("  Raw target lengths:      min=$(minimum(all_raw_lengths)), max=$(maximum(all_raw_lengths)), mean=$(round(mean(all_raw_lengths); digits=1))")
    println("  Final CTC target lengths: min=$(minimum(all_ctc_lengths)), max=$(maximum(all_ctc_lengths)), mean=$(round(mean(all_ctc_lengths); digits=1))")
end

# Save spectrogram images for visual inspection
println("\n" * "=" ^ 70)
println("PART 3: Save spectrogram PNG for visual inspection")
println("=" ^ 70)

outdir = "diagnostic_output"
mkpath(outdir)

rng2 = MersenneTwister(123)
sample = generate_sample(cfg; rng=rng2)
spec = sample.spectrogram
label = token_ids_to_label(sample.token_ids)

fig = Figure(size=(1400, 400))
ax = Axis(fig[1, 1]; title="Spectrogram (200-900 Hz, linear band) — first 1000 frames",
    xlabel="Frame", ylabel="Mel bin")
n_show = min(1000, size(spec, 2))
heatmap!(ax, 1:n_show, 1:size(spec, 1), spec[:, 1:n_show]'; colormap=:viridis)
save(joinpath(outdir, "spectrogram_diagnostic.png"), fig)
println("  Saved $outdir/spectrogram_diagnostic.png")
println("  Label: $(first(label, 120))...")

# Zoomed-in version (first 200 frames)
fig2 = Figure(size=(1400, 400))
ax2 = Axis(fig2[1, 1]; title="Spectrogram (ZOOMED, first 200 frames)",
    xlabel="Frame", ylabel="Freq bin")
n_zoom = min(200, size(spec, 2))
heatmap!(ax2, 1:n_zoom, 1:size(spec, 1), spec[:, 1:n_zoom]'; colormap=:viridis)
save(joinpath(outdir, "spectrogram_zoomed.png"), fig2)
println("  Saved $outdir/spectrogram_zoomed.png")

# Per-bin average profile (should show peaks at tone frequencies)
fig3 = Figure(size=(800, 400))
ax3 = Axis(fig3[1, 1]; title="Average bin power (should show peaks at tone freqs)",
    xlabel="Freq bin", ylabel="Mean power (log10)")
bin_avg = vec(mean(spec; dims=2))
barplot!(ax3, 1:length(bin_avg), bin_avg)
save(joinpath(outdir, "bin_profile.png"), fig3)
println("  Saved $outdir/bin_profile.png")

# Part 4: Encoder output analysis (if checkpoint provided)
if checkpoint_path !== nothing && isfile(checkpoint_path)
    println("\n" * "=" ^ 70)
    println("PART 4: Encoder + CTC head analysis (from checkpoint)")
    println("=" ^ 70)

    using JLD2

    # build_model is defined in train.jl, not in MorseDecoder; inline it here
    function build_model(n_bins::Int; dim=384, encoder_dim=nothing, n_heads=6,
            encoder_layers=6, decoder_layers=2, cross_layers=2,
            decoder_input_dropout=Float32(0.1), self_attn_residual_scale=Float32(1.0),
            qk_norm::Bool=true)
        enc_dim = something(encoder_dim, dim)
        encoder = SpectrogramEncoder(n_bins, enc_dim, n_heads, encoder_layers; qk_norm)
        decoder = SpectrogramDecoder(VOCAB_SIZE, dim, n_heads, decoder_layers;
            n_cross_layers=cross_layers, decoder_input_dropout, self_attn_residual_scale, qk_norm)
        ctc_head = Flux.Dense(enc_dim => CTC_VOCAB_SIZE)
        encoder_proj = (enc_dim == dim) ? nothing : Flux.Dense(enc_dim => dim)
        SpectrogramEncoderDecoder(encoder, decoder, ctc_head, encoder_proj)
    end

    d = jldopen(checkpoint_path, "r") do f
        (; step=f["step"], n_bins=f["n_bins"], model_state=f["model_state"],
         dim=get(f, "dim", 384), encoder_dim=get(f, "encoder_dim", nothing),
         encoder_layers=get(f, "encoder_layers", 6), n_heads=get(f, "n_heads", 6),
         decoder_layers=get(f, "decoder_layers", 4), cross_layers=get(f, "cross_layers", 6),
         decoder_input_dropout=Float32(get(f, "decoder_input_dropout", 0.1)),
         self_attn_residual_scale=Float32(get(f, "self_attn_residual_scale", 1.0)),
         qk_norm=get(f, "qk_norm", true))
    end

    enc_dim = d.encoder_dim !== nothing && d.encoder_dim != d.dim ? d.encoder_dim : nothing
    model = build_model(d.n_bins; dim=d.dim, encoder_dim=enc_dim,
        n_heads=d.n_heads, encoder_layers=d.encoder_layers,
        decoder_layers=d.decoder_layers, cross_layers=d.cross_layers,
        decoder_input_dropout=d.decoder_input_dropout,
        self_attn_residual_scale=d.self_attn_residual_scale,
        qk_norm=d.qk_norm)
    Flux.loadmodel!(model, d.model_state)
    model = device(model)
    println("  Loaded checkpoint from step $(d.step)")

    Flux.testmode!(model)

    # Generate a batch and run encoder
    rng3 = MersenneTwister(456)
    batch = generate_training_batch(cfg, 4, 512; rng=rng3)
    spec_gpu = device(batch.spectrogram)

    enc_mem, dec_mem = encode(model, spec_gpu)
    enc_cpu = cpu(enc_mem)
    ctc_logits = cpu(model.ctc_head(enc_mem))

    println("\n  Encoder output (enc_mem):")
    println("    shape:    $(size(enc_cpu))")
    println("    min:      $(minimum(enc_cpu))")
    println("    max:      $(maximum(enc_cpu))")
    println("    mean:     $(mean(enc_cpu))")
    println("    std:      $(std(enc_cpu))")

    # Check if encoder output varies across time (critical!)
    for b in 1:min(4, size(enc_cpu, 3))
        enc_b = enc_cpu[:, :, b]  # (dim, time)
        time_std = mean(std(enc_b; dims=2))  # avg std across time per feature
        feat_std = mean(std(enc_b; dims=1))  # avg std across features per time step

        # Cosine similarity between first and last frame
        f1 = enc_b[:, 1]
        fend = enc_b[:, end]
        fmid = enc_b[:, size(enc_b, 2) ÷ 2]
        cos_first_last = dot(f1, fend) / (norm(f1) * norm(fend) + 1e-8)
        cos_first_mid = dot(f1, fmid) / (norm(f1) * norm(fmid) + 1e-8)

        println("    sample $b: time_std=$(round(time_std; digits=4)), feat_std=$(round(feat_std; digits=4)), cos(f1,fmid)=$(round(cos_first_mid; digits=4)), cos(f1,fend)=$(round(cos_first_last; digits=4))")
    end

    println("\n  CTC logits:")
    println("    shape:    $(size(ctc_logits))")
    println("    min:      $(minimum(ctc_logits))")
    println("    max:      $(maximum(ctc_logits))")

    # Check: what does CTC predict? Is blank dominant?
    for b in 1:min(4, size(ctc_logits, 3))
        logits_b = ctc_logits[:, :, b]  # (CTC_VOCAB_SIZE, time)
        preds = vec(mapslices(argmax, logits_b; dims=1))
        n_blank = count(==(CTC_BLANK_IDX), preds)
        n_total = length(preds)
        unique_preds = sort(unique(preds))
        println("    sample $b: $(n_blank)/$(n_total) blank ($(round(100*n_blank/n_total; digits=1))%), unique predictions: $(length(unique_preds)), values: $(first(unique_preds, 10))")
    end

    # Softmax distribution analysis — how peaked/flat are the CTC logits?
    for b in 1:min(2, size(ctc_logits, 3))
        logits_b = ctc_logits[:, :, b]
        probs_b = softmax(logits_b; dims=1)
        max_probs = vec(maximum(probs_b; dims=1))
        entropy = -vec(sum(probs_b .* log.(probs_b .+ 1f-10); dims=1))
        println("    sample $b softmax: max_prob mean=$(round(mean(max_probs); digits=4)), entropy mean=$(round(mean(entropy); digits=3)) (uniform=$(round(log(size(ctc_logits,1)); digits=3)))")
    end
else
    println("\n  Skipping encoder analysis (no checkpoint). Use --checkpoint path/to/checkpoint.jld2")
end
