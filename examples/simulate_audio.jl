#!/usr/bin/env julia
# Simulate a multi-station Morse band; save WAV, transcript, and spectrogram PNG.
# Run from repo root: julia --project=. examples/simulate_audio.jl [output_prefix]
# Default prefix "band" → band.wav, band_transcript.txt, band.png

using MorseDecoder, Random, WAV, CairoMakie

function main()
    prefix = length(ARGS) >= 1 ? ARGS[1] : "band"
    rng = MersenneTwister(42)

    # Random band: 3 stations, 8 kHz (use fixed seed for reproducibility)
    scene = random_band(rng; n_stations=3, sr=8000)

    # Spectrogram config: 200–800 Hz band, nfft/hop tuned for model input
    # (enough time/freq resolution, compact so the heatmap array stays manageable)
    spec_cfg = SpectrogramConfig(; freq_lo=200f0, freq_hi=800f0)  # default nfft=1024 for ~10 Hz resolution

    # Save audio
    wavpath = "$prefix.wav"
    wavwrite(scene.audio, wavpath; Fs=scene.sr)
    println("Wrote audio: $wavpath ($(length(scene.audio)/scene.sr) s @ $(scene.sr) Hz)")

    # Save transcript (one line per station)
    txtpath = "$(prefix)_transcript.txt"
    open(txtpath, "w") do io
        for (i, (st, txt)) in enumerate(zip(scene.stations, scene.texts))
            line = "Station $i @ $(round(st.frequency; digits=1)) Hz (WPM $(st.wpm)): $txt"
            println(io, line)
            println(line)
        end
    end
    println("Wrote transcript: $txtpath")

    # Spectrogram visualization (200–800 Hz, same as model input)
    fig = plot_band(scene, spec_cfg)
    pngpath = "$prefix.png"
    save(pngpath, fig)
    println("Wrote spectrogram: $pngpath")
end

main()
