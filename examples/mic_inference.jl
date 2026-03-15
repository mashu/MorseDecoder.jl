#!/usr/bin/env julia
# Run inference from the default microphone for a fixed number of chunks, then decode.
# Requires a trained checkpoint (same spectrogram setup: 40 bins, 44.1 kHz).
#
#   julia --project=. examples/mic_inference.jl [--checkpoint checkpoints/checkpoint_latest.jld2] [--gpu] [--chunks 5] [--chunk-seconds 1.0]
#
# Use --gpu to run the model on CUDA. Press Ctrl+C to stop if you use a large --chunks.

using ArgParse
if "--gpu" in ARGS
    using CUDA
end
using MorseDecoder
using Flux
using JLD2

function parse_commandline()
    s = ArgParseSettings(description = "Decode CW from microphone using a trained checkpoint.")
    @add_arg_table! s begin
        "--checkpoint"
        help = "Path to checkpoint JLD2 (model_state, n_bins, dim, ...)"
        arg_type = String
        default = "checkpoints/checkpoint_latest.jld2"
        "--gpu"
        help = "Run model on GPU"
        action = :store_true
        "--chunks"
        help = "Number of spectrogram chunks to record (each chunk_seconds long)"
        arg_type = Int
        default = 5
        "--chunk-seconds"
        help = "Duration of each chunk in seconds"
        arg_type = Float64
        default = 1.0
        "--sample-rate"
        help = "Microphone sample rate (must match training, e.g. 44100)"
        arg_type = Int
        default = 44100
    end
    parse_args(ARGS, s)
end

function load_checkpoint(path::String)
    isfile(path) || return nothing
    jldopen(path, "r") do f
        step = f["step"]
        n_bins = f["n_bins"]
        model_state = f["model_state"]
        dim = haskey(f, "dim") ? f["dim"] : 384
        encoder_dim = haskey(f, "encoder_dim") ? f["encoder_dim"] : nothing
        encoder_layers = haskey(f, "encoder_layers") ? f["encoder_layers"] : 6
        n_heads = haskey(f, "n_heads") ? f["n_heads"] : 6
        decoder_layers = haskey(f, "decoder_layers") ? f["decoder_layers"] : 2
        cross_layers = haskey(f, "cross_layers") ? f["cross_layers"] : 2
        decoder_input_dropout = haskey(f, "decoder_input_dropout") ? Float32(f["decoder_input_dropout"]) : 0.1f0
        self_attn_residual_scale = haskey(f, "self_attn_residual_scale") ? Float32(f["self_attn_residual_scale"]) : 1.0f0
        (; step, n_bins, model_state, dim, encoder_dim, encoder_layers, n_heads, decoder_layers, cross_layers, decoder_input_dropout, self_attn_residual_scale)
    end
end

function build_model(n_bins::Int; dim=384, encoder_dim=nothing, n_heads=6, encoder_layers=6, decoder_layers=2, cross_layers=2, decoder_input_dropout=Float32(0.1), self_attn_residual_scale=Float32(1.0))
    enc_dim = something(encoder_dim, dim)
    encoder = SpectrogramEncoder(n_bins, enc_dim, n_heads, encoder_layers)
    decoder = SpectrogramDecoder(VOCAB_SIZE, dim, n_heads, decoder_layers;
        n_cross_layers=cross_layers, decoder_input_dropout, self_attn_residual_scale)
    ctc_head = Dense(enc_dim => CTC_VOCAB_SIZE)
    encoder_proj = (enc_dim == dim) ? nothing : Dense(enc_dim => dim)
    SpectrogramEncoderDecoder(encoder, decoder, ctc_head, encoder_proj)
end

function main()
    args = parse_commandline()
    device = args["gpu"] ? gpu : cpu

    # Load model from checkpoint
    d = load_checkpoint(args["checkpoint"])
    if d === nothing
        @error "Checkpoint not found" path=args["checkpoint"]
        return
    end
    model = build_model(d.n_bins; dim=d.dim, encoder_dim=d.encoder_dim, n_heads=d.n_heads,
        encoder_layers=d.encoder_layers, decoder_layers=d.decoder_layers, cross_layers=d.cross_layers,
        decoder_input_dropout=d.decoder_input_dropout, self_attn_residual_scale=d.self_attn_residual_scale)
    Flux.loadmodel!(model, d.model_state)
    model = device(model)
    @info "Model loaded" checkpoint=args["checkpoint"] step=d.step n_bins=d.n_bins

    # Mic: same sample rate and bin count as training (skip opening mic if --chunks 0)
    n_chunks = args["chunks"]
    if n_chunks <= 0
        @info "Skipping mic (--chunks 0); model loaded successfully."
        return
    end
    sr = args["sample-rate"]
    cfg = mic_spectrogram_config(sr)
    stream = open_mic_input(; sample_rate=sr)
    src = MicSpectrogramSource(stream, cfg; chunk_seconds=args["chunk-seconds"])
    @info "Recording" chunks=n_chunks chunk_seconds=args["chunk-seconds"] sample_rate=sr
    chunks = collect(Iterators.take(src, n_chunks))
    close(stream)
    isempty(chunks) && (@warn "No chunks recorded"; return)

    # Decode (chunk-by-chunk conversation)
    decoded_ids = decode_conversation(model, chunks, device)
    text = token_ids_to_label(decoded_ids)
    @info "Decoded" n_tokens=length(decoded_ids) text=isempty(text) ? "(empty)" : text
end

main()
