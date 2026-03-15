# MorseDecoder.jl

**MorseDecoder.jl** is a training pipeline for a spectrogram encoder–decoder that decodes multi-station CW (Morse) signals. It uses [MorseSimulator.jl](https://github.com/mashu/MorseSimulator.jl) for data: mel spectrograms in the **200–900 Hz** band with ~**10 Hz** frequency resolution (to separate signals) and time resolution suitable for **50 WPM** (dots vs dashes). Labels use the simulator's special tokens: `<START>`, `<END>`, `[S1]`…`[S6]`, `[TS]`, `[TE]`.

## Features

- **Chunked training**: Conversations split at `[TS]`/`[TE]` boundaries so no chunk cuts mid-turn.
- **Encoder–decoder**: Full self-attention over time in the encoder; causal self-attention + cross-attention in the decoder; optional CTC head for streaming.
- **DirectPath data**: Analytic mel spectrograms from MorseSimulator (no audio waveform), with WPM, amplitude, SNR, overlap, QSB fading, and per-station frequency.
- **Decoding**: Autoregressive decode, CTC greedy decode, and chunk-by-chunk `decode_conversation` for streaming.

## Documentation

- [Installation](installation.md)
- [Quick start](quickstart.md)
- [API reference](api.md)

## License

See `Project.toml` for authors and version.
