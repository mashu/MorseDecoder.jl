#!/usr/bin/env julia
# Generate one training-style sample with MorseSimulator: WAV, transcript, spectrogram PNG.
# The PNG shows exactly what the train encoder gets: full (unchunked) spectrogram with red
# lines at chunk boundaries (same chunking as training, max_frames=512).
#
# Run from repo root:
#   julia --project=. examples/simulate_audio.jl [output_prefix]
#
# Output (default prefix "band"):
#   band.wav             — mixed audio (44.1 kHz)
#   band_transcript.txt  — training label (flat_text format)
#   band.png             — encoder input: full spectrogram + red chunk boundaries

using MorseDecoder
using MorseSimulator: DatasetConfig, DirectPath, generate_sample_with_audio, save_audio
using Random
using CairoMakie

const MAX_FRAMES = 512  # same as training default (chunk size)

function main()
    prefix = length(ARGS) >= 1 ? ARGS[1] : "band"
    seed = 42
    rng = MersenneTwister(seed)

    config = DatasetConfig(;
        path = DirectPath(),
        sample_rate = 44100,
        fft_size = 4096,
        hop_size = 128,
        f_min = 200.0,
        f_max = 900.0,
        stations = 2:4,
    )

    # MorseDecoder Sample (spectrogram + token_ids + token_timing) for chunking
    sample = generate_sample(config; rng)
    # Same seed → same conversation for WAV
    _, audio = generate_sample_with_audio(MersenneTwister(seed), config)

    # Save audio
    wavpath = "$prefix.wav"
    save_audio(wavpath, audio)
    println("Wrote audio: $wavpath")

    # Transcript (training label)
    txtpath = "$(prefix)_transcript.txt"
    open(txtpath, "w") do io
        println(io, "Training label (flat_text):")
        println(io, token_ids_to_label(sample.token_ids))
    end
    println("Wrote transcript: $txtpath")

    # Encoder input: full spectrogram (unchunked) with red lines where training chunks split
    chunks_tc = chunk_conversation(sample, MAX_FRAMES)
    boundary_frames = [1 + sum(size(chunks_tc[j].spectrogram, 2) for j in 1:i) for i in 1:length(chunks_tc)-1]

    spec_cfg = SpectrogramConfig(; nfft=4096, hop=128, freq_lo=200f0, freq_hi=900f0)
    fig = plot_encoder_input(
        sample.spectrogram,
        config.sample_rate,
        spec_cfg;
        title="Encoder input (full conversation, chunk boundaries)",
        chunk_boundaries=isempty(boundary_frames) ? nothing : boundary_frames,
    )
    pngpath = "$prefix.png"
    save(pngpath, fig)
    println("Wrote spectrogram: $pngpath (red lines = chunk boundaries, max_frames=$MAX_FRAMES)")
end

main()
