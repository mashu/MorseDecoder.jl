#!/usr/bin/env julia
# Show ground-truth transcripts per sequence, with all chunks shown as chunk1="...", chunk2="...", ...
# Each sequence is one full conversation split into chunks (by [TS]/[TE] and max_frames).
#
# Run from repo root: julia --project=. examples/show_batch_transcripts.jl [--sequences 3] [--max-frames 512]

using ArgParse
using Random
using MorseSimulator: DatasetConfig, DirectPath
using MorseDecoder: generate_sample, chunk_conversation, token_ids_to_label

function parse_commandline()
    s = ArgParseSettings(description = "Print transcripts per sequence with all chunks (chunk1=..., chunk2=..., ...).")
    @add_arg_table! s begin
        "--sequences"
        help = "Number of sequences (full conversations) to show"
        arg_type = Int
        default = 3
        "--max-frames"
        help = "Max frames per chunk (same as training)"
        arg_type = Int
        default = 512
        "--seed"
        help = "Random seed"
        arg_type = Int
        default = 42
    end
    ArgParse.parse_args(ARGS, s)
end

function sampler_config()
    DatasetConfig(; path = DirectPath(), sample_rate = 44100, fft_size = 4096, hop_size = 128,
        f_min = 200.0, f_max = 900.0, stations = 2:4)
end

function main()
    args = parse_commandline()
    rng = MersenneTwister(args["seed"])
    cfg = sampler_config()
    n_seq = args["sequences"]
    max_frames = args["max-frames"]

    println("batch 1:")
    for s in 1:n_seq
        sample = generate_sample(cfg; rng)
        chunks = chunk_conversation(sample, max_frames)
        parts = String[]
        for (c, ch) in enumerate(chunks)
            txt = isempty(ch.token_ids) ? "(empty)" : token_ids_to_label(ch.token_ids)
            escaped = replace(txt, '"' => "\\\"")
            push!(parts, "chunk$(c)=\"$(escaped)\"")
        end
        println("seq$(s): ", join(parts, ", "))
    end
end

main()
