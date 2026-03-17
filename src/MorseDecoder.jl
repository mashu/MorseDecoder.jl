"""
    MorseDecoder

Training pipeline for a spectrogram encoder–decoder that decodes multi-station
CW Morse. Uses MorseSimulator.jl for data: 200–900 Hz mel spectrograms with
~10 Hz resolution (to separate signals) and time resolution suitable for 50 WPM.
Labels use simulator tokens: <START>, <END>, [S1]..[S6], [TS], [TE].

# Quick start

```julia
using MorseDecoder, MorseSimulator, Random

cfg = DatasetConfig(; path = DirectPath(), sample_rate = 44100, fft_size = 4096, hop_size = 128,
    n_mels = 40, f_min = 200.0, f_max = 900.0, stations = 2:4)   # 200–900 Hz, ~10 Hz resolution
rng = MersenneTwister(42)
s = generate_sample(cfg; rng)   # one full conversation
batch = generate_training_batch(cfg, 32, 512; rng)   # one batch of chunks
```

# Chunked training and decoding

- **ChunkedConversation(sample, max_frames)** — iterable over `Sample` chunks of one conversation (split at [TS]/[TE]).
- **BatchIterator(cfg, batch_size, max_frames; rng)** — infinite iterator of `Batch` for training.
- **decode_conversation(model, chunks, to_device; ...)** — decode a full conversation from an iterable of spectrogram chunks (chunk-by-chunk with continuation; supports streaming).
"""
module MorseDecoder

using Random
using FFTW: rfft, plan_rfft
using CairoMakie

# ── Source files (order matters) ─────────────────────────────────────────────

include("morse.jl")           # ALPHABET, NUM_CHARS, CHAR_TO_IDX, IDX_TO_CHAR
include("vocab.jl")           # token IDs, CTC constants
include("spectrogram.jl")     # STFT for inference on real audio
include("audio.jl")           # Live mic input → spectrogram chunks for decode_conversation
include("sampler.jl")         # Sample, Batch (must be before model.jl)
include("model.jl")           # encoder–decoder architecture (Onion + Flux)
include("loss.jl")            # sequence_cross_entropy
include("training.jl")        # prepare_decoder_batch, train_step, CTC targets
include("decode.jl")          # autoregressive decode, CTC greedy decode
include("viz.jl")             # plots

# ── Public API ───────────────────────────────────────────────────────────────

export
    # Morse alphabet (for vocab / inference)
    MORSE_TABLE, ALPHABET, NUM_CHARS,
    CHAR_TO_IDX, IDX_TO_CHAR, encode_text, decode_indices,

    # Vocab (simulator label ↔ token IDs)
    label_to_token_ids, token_ids_to_label, token_ids_to_plain_text,
    VOCAB_SIZE, START_TOKEN_IDX, PAD_TOKEN_IDX, EOS_TOKEN_IDX,
    SPEAKER_1_IDX, speaker_token_id, is_speaker_token,
    TS_TOKEN_IDX, TE_TOKEN_IDX,

    # CTC vocabulary constants
    CTC_VOCAB_SIZE, CTC_BLANK_IDX,

    # Spectrogram (inference on WAV)
    SpectrogramConfig, compute_spectrogram, num_bins, num_frames,
    spectrogram_to_model_scale,

    # Sampler (MorseSimulator)
    Sample, generate_sample,
    Batch, collate,
    transmission_segments, chunk_conversation, generate_training_batch,
    ChunkedConversation, BatchIterator,

    # Visualization
    plot_spectrogram, plot_chunk,

    # Model architecture
    SpectrogramEncoder, SpectrogramDecoder, SpectrogramEncoderDecoder,
    encode, ENCODER_DOWNSAMPLE,

    # Loss
    sequence_cross_entropy,

    # Training
    prepare_decoder_batch, prepare_training_batch, train_step,
    prepare_ctc_targets,

    # Decoding
    decode_autoregressive, decode_conversation,
    ctc_greedy_decode,

    # Live audio (mic) for inference
    mic_spectrogram_config, MicSpectrogramSource, open_mic_input

end # module MorseDecoder
