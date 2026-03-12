#!/usr/bin/env julia
# Generate one training-style sample with MorseSimulator: WAV, transcript, spectrogram PNG.
#
# Run from repo root:
#   julia --project=. examples/simulate_audio.jl [output_prefix]
#
# Output (default prefix "band"):
#   band.wav             — mixed audio (44.1 kHz)
#   band_transcript.txt  — training label (flat_text format)
#   band.png             — mel spectrogram (200–900 Hz)

using MorseDecoder
using MorseSimulator: DatasetConfig, DirectPath, generate_sample_with_audio, save_audio
using Random
using CairoMakie

function main()
    prefix = length(ARGS) >= 1 ? ARGS[1] : "band"
    rng = MersenneTwister(42)

    config = DatasetConfig(;
        path = DirectPath(),
        sample_rate = 44100,
        fft_size = 4096,
        hop_size = 256,
        n_mels = 40,
        f_min = 200.0,
        f_max = 900.0,
        stations = 2:4,
    )
    sample, audio = generate_sample_with_audio(rng, config)

    # Save audio
    wavpath = "$prefix.wav"
    save_audio(wavpath, audio)
    println("Wrote audio: $wavpath")

    # Transcript (training label)
    txtpath = "$(prefix)_transcript.txt"
    open(txtpath, "w") do io
        println(io, "Training label (flat_text):")
        println(io, sample.label)
    end
    println("Wrote transcript: $txtpath")

    # Spectrogram plot (200–900 Hz mel; cfg for axis labels)
    spec_cfg = SpectrogramConfig(; hop=256, freq_lo=200f0, freq_hi=900f0)
    fig = plot_spectrogram(Float32.(sample.mel_spectrogram), config.sample_rate, spec_cfg; title="Mel 200–900 Hz")
    pngpath = "$prefix.png"
    save(pngpath, fig)
    println("Wrote spectrogram: $pngpath")
end

main()
