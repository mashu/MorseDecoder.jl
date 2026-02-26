"""
    MorseDecoder

Morse data pipeline: simulation, spectrogram, WAV I/O, dataset generation.
No neural network model — data exploration and preprocessing only; modeling lives in the notebook.

# Layout
- **alphabet**     — Morse table, character ↔ index
- **audio/**       — spectrogram (CPU), WAV I/O, visualization
- **simulation/**  — types, simulator, contests, drift (Morse signal generation)
- **data/**        — dataset manifest, generation, loading
"""

module MorseDecoder

using Random
using Statistics
using DSP
using AbstractFFTs: rfft
using FFTW
using CairoMakie

# ── Shared ───────────────────────────────────────────────────────────────────
include("alphabet.jl")

# ── Audio ─────────────────────────────────────────────────────────────────────
include("audio/spectrogram.jl")
include("audio/wav.jl")

# ── Simulation ───────────────────────────────────────────────────────────────
include("simulation/types.jl")
include("simulation/simulator.jl")
include("simulation/callsigns.jl")
include("simulation/contests.jl")
include("simulation/drift.jl")

# ── Data ─────────────────────────────────────────────────────────────────────
include("data/dataset.jl")

# ── Audio visualization ──────────────────────────────────────────────────────
include("audio/visualization.jl")

# ── Public API ───────────────────────────────────────────────────────────────
export
    MORSE_TABLE, ALPHABET, NUM_CHARS, NUM_TAGS, CHAR_TO_IDX, IDX_TO_CHAR,
    Modulation, NoiseModel, KeyingStyle,
    SineModulation, WhiteGaussianNoise, NoNoise,
    HumanKeying, PerfectKeying, DriftingKeying,
    MorseSignalConfig, MorseSignal,
    generate_signal, random_config, random_morse_text,
    MorseFFTConfig, audio_to_tokens, n_bins, bin_range, compute_spectrogram,
    random_callsign, PrefixDef, PREFIX_POOL,
    AbstractContest, ContestExchange,
    CQWorldWide, CQWPX, ARRLDX, ARRLSweepstakes, Sprint, GeneralQSO, IARUHF,
    generate_exchange, random_exchange,
    generate_contest_signal, sample_word_wpms,
    DatasetConfig, generate_dataset, load_dataset, DatasetLoader, num_freq_bins,
    load_sample, load_batch, random_training_batch, collate_batch,
    write_wav, read_wav, ManifestEntry, dataset_stats,
    read_manifest, write_manifest,
    plot_waveform

end # module MorseDecoder
