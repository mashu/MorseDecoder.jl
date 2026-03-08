#!/usr/bin/env julia
# Train the spectrogram encoder–decoder. Run from repo root:
#   julia --project=. examples/train.jl [--gpu] [--steps 500]
#
# Uses MorseDecoder sampler for batches; one gradient step per batch.
# For CUDA: pass --gpu and have CUDA.jl + Flux.gpu available.

using MorseDecoder
using Flux
using Random
using CUDA
using cuDNN

function parse_args()
    gpu = "--gpu" in ARGS
    steps = 500
    for i in eachindex(ARGS)
        ARGS[i] == "--steps" && i < length(ARGS) && (steps = parse(Int, ARGS[i + 1]); break)
    end
    (; gpu, steps)
end

function build_model(n_bins::Int, chunk_frames::Int; dim=128, n_heads=4, n_layers=2)
    token_dim = n_bins * chunk_frames
    encoder = SpectrogramEncoder(token_dim, dim, n_heads, n_layers, chunk_frames)
    decoder = SpectrogramDecoder(VOCAB_SIZE, dim, n_heads, n_layers)
    SpectrogramEncoderDecoder(encoder, decoder)
end

function main()
    args = parse_args()
    rng = MersenneTwister(42)

    # Data: 1–3 stations, same spectrogram config as in SamplerConfig
    cfg = SamplerConfig(; n_stations_range=1:3, spec=SpectrogramConfig())
    batch = generate_batch(cfg, 8; rng)

    n_bins = size(batch.spectrogram, 1)
    chunk_frames = 4
    model = build_model(n_bins, chunk_frames)

    # Device first, then optimiser so state lives on same device as params (avoids scalar indexing on GPU)
    device = args.gpu ? gpu : cpu
    if args.gpu
        model = device(model)
    end
    opt = Flux.setup(Adam(1f-4), model)

    println("Training for $(args.steps) steps on $(args.gpu ? "GPU" : "CPU") (batch_size=$(size(batch.spectrogram, 2)))")

    for step in 1:args.steps
        batch = generate_batch(cfg, 8; rng)
        spec, decoder_input, decoder_target, station_mask = prepare_training_batch(batch)
        spec = device(spec)
        decoder_input = device(decoder_input)
        decoder_target = device(decoder_target)
        station_mask = device(station_mask)

        result = Flux.withgradient(model) do m
            train_step(m, spec, decoder_input, decoder_target, station_mask)
        end
        Flux.update!(opt, model, result.grad[1])
        loss = result.val

        step % 50 == 0 && println("  step $step  loss = $(round(loss; digits=4))")
    end

    # Quick decode test on last batch (one sample, 2 stations)
    println("\nDecode test (one spectrogram, 2 stations):")
    test_spec = batch.spectrogram[:, 1:1, :]   # (n_bins, 1, time)
    test_spec = device(test_spec)
    ids = decode_autoregressive(model, test_spec, 2; max_len=32)
    ids_cpu = cpu(ids)
    for k in 1:2
        seq = [ids_cpu[i, k] for i in 1:size(ids_cpu, 1)]
        # Clip to valid vocab and decode (ignore START/PAD)
        chars = [i in 1:NUM_CHARS ? IDX_TO_CHAR[i] : '?' for i in seq]
        println("  station $k: ", String(chars))
    end
end

main()
