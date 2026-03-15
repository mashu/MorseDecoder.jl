"""
    MorseDecoder

Training pipeline for a spectrogram encoder–decoder that decodes multi-station
CW Morse. Uses MorseSimulator.jl for data: 200–900 Hz mel spectrograms with
~10 Hz resolution (to separate signals) and time resolution suitable for 50 WPM.
Labels use simulator tokens: <START>, <END>, [S1]..[S6], [TS], [TE].

# Quick start

```julia
using MorseDecoder, Random

rng = MersenneTwister(42)
cfg = SamplerConfig()   # 200–900 Hz, ~10 Hz resolution
s   = generate_sample(cfg; rng)   # one full conversation
batch = generate_chunked_batch(cfg, 32, rng; max_frames=512)   # one batch of chunks
```

# Chunked training and decoding

- **ChunkedConversation(spec, token_ids, max_frames)** — iterable over `Sample` chunks of one conversation (split at [TS]/[TE]).
- **ChunkedSampleSource(cfg, max_frames; rng)** — infinite iterator of chunked samples.
- **ChunkedBatchSource(cfg, batch_size, max_frames; rng)** — infinite iterator of `Batch` for training.
- **decode_conversation(model, chunks, to_device; ...)** — decode a full conversation from an iterable of spectrogram chunks (chunk-by-chunk with continuation; supports streaming).
"""
module MorseDecoder

using Random
using FFTW: rfft, plan_rfft
using CairoMakie
using CTCLoss  # model.jl calls CTCLoss.ctc_loss_batched, CTCLoss.ctc_greedy_decode

# ── Source files (order matters) ─────────────────────────────────────────────

include("morse.jl")           # ALPHABET, NUM_CHARS, CHAR_TO_IDX, IDX_TO_CHAR
include("vocab.jl")           # token IDs for simulator labels (<START>, [S1]..[S6], [TS], [TE])
include("spectrogram.jl")     # STFT for inference on real audio
include("audio.jl")           # Live mic input → spectrogram chunks for decode_conversation
include("sampler.jl")          # Sample, Batch (must be before model.jl)
include("model.jl")           # encoder–decoder (Onion + Flux)
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

    # Spectrogram (inference on WAV)
    SpectrogramConfig, compute_spectrogram, num_bins, num_frames,
    spectrogram_to_model_scale,

    # Sampler (MorseSimulator)
    Sample, SamplerConfig, generate_sample,
    Batch, collate, generate_batch, generate_batch_fast,
    transmission_segments, segments_to_chunks, chunked_samples, generate_chunked_batch,
    ChunkedConversation, ChunkedSampleSource, ChunkedBatchSource,

    # Visualization
    plot_spectrogram, plot_chunk,

    # Model
    SpectrogramEncoder, SpectrogramDecoder, SpectrogramEncoderDecoder,
    encode, ENCODER_DOWNSAMPLE,
    prepare_decoder_batch, prepare_training_batch, train_step,
    decode_autoregressive, decode_conversation, sequence_cross_entropy,

    # CTC (uses CTCLoss.jl; blank = CTC_BLANK_IDX)
    CTC_VOCAB_SIZE, CTC_BLANK_IDX,
    prepare_ctc_targets, ctc_greedy_decode,

    # Live audio (mic) for inference
    mic_spectrogram_config, MicSpectrogramSource, open_mic_input

end # module MorseDecoder
