#!/usr/bin/env julia
# Simulate **contest-style** band: one runner (station 1) calling CQ and working
# multiple responders (stations 2, 3, …) in sequence. Saves WAV, transcript, and
# spectrogram PNG.
#
# Run from repo root:
#   julia --project=. examples/simulate_audio.jl [output_prefix]
#
# Output (default prefix "band"):
#   band.wav             — mixed audio (8 kHz), stations key in turn
#   band_transcript.txt  — per-station full text + interleaved "[1] ... [2] ..."
#   band.png             — waveform + spectrogram with station frequencies

using MorseDecoder, Random, WAV, CairoMakie

function main()
    prefix = length(ARGS) >= 1 ? ARGS[1] : "band"
    rng = MersenneTwister(42)

    # Contest-style: one runner calling CQ and working 2–4 responders in sequence.
    # Optional overlap: hunters sometimes start before runner finishes "CQ ... K".
    responder_overlap_ms = rand(rng) < 0.5 ? 80f0 + 120f0 * rand(rng) : 0f0  # 0 or ~80–200 ms
    scene = random_contest_conversation_band(rng; n_responders=2 + rand(rng, 0:2), sr=8000, responder_overlap_ms=responder_overlap_ms)
    spec_cfg = SpectrogramConfig(; freq_lo=100f0, freq_hi=900f0)  # same as train.jl

    # Save audio
    wavpath = "$prefix.wav"
    wavwrite(scene.audio, wavpath; Fs=scene.sr)
    duration_s = length(scene.audio) / scene.sr
    println("Wrote audio: $wavpath ($(round(duration_s; digits=2)) s @ $(scene.sr) Hz)")

    # Transcript: per-station lines + one interleaved line (turn order = model target [1] ... [2] ... [1] ...)
    txtpath = "$(prefix)_transcript.txt"
    interleaved = if scene.turns !== nothing
        join(("[$(spk)] $(txt)" for (spk, txt) in scene.turns), " ")
    else
        join(("[$(k)] $(scene.texts[k])" for k in 1:length(scene.texts)), " ")
    end
    open(txtpath, "w") do io
        println(io, "Contest-style: 1 runner + $(length(scene.stations)-1) responders, turn-based.")
        println(io, "")
        for (i, (st, txt)) in enumerate(zip(scene.stations, scene.texts))
            line = "Station $i @ $(round(st.frequency; digits=1)) Hz (WPM $(round(st.wpm; digits=0))): $txt"
            println(io, line)
            println(line)
        end
        println(io, "")
        println(io, "Interleaved (model target format):")
        println(io, interleaved)
        println()
        println("Interleaved (model target): ", interleaved)
    end
    println("Wrote transcript: $txtpath")

    # Spectrogram plot (100–900 Hz, same as model input)
    fig = plot_band(scene, spec_cfg)
    pngpath = "$prefix.png"
    save(pngpath, fig)
    println("Wrote spectrogram: $pngpath")
end

main()
