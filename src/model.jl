"""
    model.jl — Spectrogram encoder–decoder architecture (Onion + Flux).

- Single-stream decoder: one output sequence per spectrogram, with speaker tokens
  (SPEAKER_1 … SPEAKER_6) so turn order is preserved. No fixed N slots; decode until EOS.
- Encoder: full self-attention over time. Decoder: causal self-attention + cross-attention to encoder.
- Optional CTC head on encoder output for streaming / joint CTC-attention training.
- Caller moves data to GPU; architecture is device-agnostic.
"""

using Flux
using KernelAbstractions: KernelAbstractions as KA, get_backend, synchronize, @kernel, @index, @uniform
using Onion
using Onion: TransformerBlock, RoPE, RMSNorm
using Einops: rearrange, @einops_str
using Random

# ─── CW front-end (CWFeatureExtractor) ─────────────────────────────────────
#
# Full mel in; one stride-2 downsample → T/2 so transformer gets ~4 frames/dot at 50 WPM.
# Two convs: conv7 → gelu, conv4 stride-2 → gelu. No BatchNorm.

"""
    CWFeatureExtractor(n_freq_bins; d_model=128)

Minimal conv front-end: (T, n_freq_bins, B) → (T/2, d_model, B).
Stack: conv7 → gelu, conv4 stride2 → gelu.
"""
struct CWFeatureExtractor{C1,C2}
    conv1::C1
    conv2::C2
end
Flux.@layer CWFeatureExtractor

function CWFeatureExtractor(n_freq_bins::Int; d_model::Int=128)
    CWFeatureExtractor(
        Conv((7,), n_freq_bins => 64; pad=3),
        Conv((4,), 64 => d_model; stride=2, pad=1),
    )
end

function (m::CWFeatureExtractor)(x)
    h = m.conv1(x) |> gelu   # (T, 64, B)
    m.conv2(h) |> gelu       # (T/2, d_model, B)
end

# ─── Encoder ───────────────────────────────────────────────────────────────

"""Encoder time is spectrogram time ÷ this (CW front-end has one stride-2 → T/2)."""
const ENCODER_DOWNSAMPLE = 2

const CW_FRONTEND_DIM = 128

"""
    SpectrogramEncoder(n_freq_bins, dim, n_heads, n_layers; ...)

Maps spectrogram to encoder hidden states. **Frontend:** CWFeatureExtractor
(conv7 → gelu, conv4 stride-2 → gelu; one downsampling → T/2). **Then:** transformer blocks.

**Time resolution:** Spectrogram ~344 Hz (hop 128 @ 44.1 kHz); after 2× downsampling the
transformer sees ~172 Hz. At 50 WPM a dot ≈ 24 ms → ~4 frames per dot.

- Input: (freq_bins, batch, time) spectrogram in log10 scale.
- Output: (dim, num_tokens, batch) where num_tokens = time ÷ 2.
"""
struct SpectrogramEncoder{F,P,B,N,R}
    frontend::F
    proj::P       # 128 → dim when dim != 128; identity-like Nothing otherwise
    blocks::B
    norm::N
    rope::R
end

Flux.@layer SpectrogramEncoder

function SpectrogramEncoder(
    n_freq_bins::Int,
    dim::Int,
    n_heads::Int,
    n_layers::Int;
    n_kv_heads::Int = n_heads,
    ff_mult::Int = 4,
    norm_eps::AbstractFloat = 1f-5,
    max_len::Int = 4096,
)
    frontend = CWFeatureExtractor(n_freq_bins; d_model=CW_FRONTEND_DIM)
    proj = dim == CW_FRONTEND_DIM ? nothing : Dense(CW_FRONTEND_DIM => dim)
    blocks = [TransformerBlock(dim, n_heads, n_kv_heads, dim * ff_mult; norm_eps) for _ in 1:n_layers]
    norm = RMSNorm(dim; eps=norm_eps)
    rope = RoPE(dim ÷ n_heads, max_len)
    SpectrogramEncoder(frontend, proj, blocks, norm, rope)
end

"""Apply encoder projection (Dense) when dims differ."""
function apply_encoder_proj(proj::Dense, h)
    T_enc, B = size(h, 2), size(h, 3)
    out = proj(reshape(h, size(h, 1), :))
    reshape(out, size(out, 1), T_enc, B)
end

"""No-op when encoder dim == decoder dim."""
apply_encoder_proj(::Nothing, h) = h

function (enc::SpectrogramEncoder)(spec::AbstractArray{T,3}) where T
    x = permutedims(spec, (3, 1, 2))   # (time, freq_bins, batch)
    h = enc.frontend(x)                 # (T/2, 128, batch)
    h = permutedims(h, (2, 1, 3))      # (128, T/2, batch)
    h = apply_encoder_proj(enc.proj, h)
    n_tokens = size(h, 2)
    rope = enc.rope[1:n_tokens]
    for block in enc.blocks
        h = block(h; rope=rope, krope=rope)
    end
    enc.norm(h)
end

# ─── Decoder (causal self-attn + cross-attn) ──────────────────────────────────

"""
    SpectrogramDecoder(vocab_size, dim, n_heads, n_decoder_layers; ...)

Single-stream autoregressive decoder: causal self-attention then cross-attention to encoder.

**Layout (Whisper-style):**
- Decoder layers are interleaved: each [decoder self-attn] → [cross-attn to encoder].
- If n_cross_layers > n_decoder_layers, extra cross-only layers run after.
- n_decoder_layers=0 → cross-only decoder (Whisper-Turbo style).
"""
struct SpectrogramDecoder{E,D,SB,CB,N,H,R}
    embed::E
    embed_dropout::D
    self_blocks::SB
    cross_blocks::CB
    norm::N
    head::H
    rope::R
    self_attn_residual_scale::Float32
end

Flux.@layer SpectrogramDecoder

function SpectrogramDecoder(
    vocab_size::Int,
    dim::Int,
    n_heads::Int,
    n_decoder_layers::Int;
    n_cross_layers::Int = n_decoder_layers,
    decoder_input_dropout::AbstractFloat = 0.0f0,
    self_attn_residual_scale::AbstractFloat = 1.0f0,
    n_kv_heads::Int = n_heads,
    ff_mult::Int = 4,
    norm_eps::AbstractFloat = 1f-5,
    max_len::Int = 2048,
)
    n_cross_layers >= n_decoder_layers || throw(ArgumentError("n_cross_layers ($n_cross_layers) must be >= n_decoder_layers ($n_decoder_layers)"))
    (n_decoder_layers > 0 || n_cross_layers >= 1) || throw(ArgumentError("when n_decoder_layers=0, n_cross_layers must be >= 1"))
    embed = Flux.Embedding(vocab_size => dim)
    embed_dropout = Flux.Dropout(decoder_input_dropout)
    self_blocks = [TransformerBlock(dim, n_heads, n_kv_heads, dim * ff_mult; norm_eps) for _ in 1:n_decoder_layers]
    cross_blocks = [TransformerBlock(dim, n_heads, n_kv_heads, dim * ff_mult; norm_eps) for _ in 1:n_cross_layers]
    norm = RMSNorm(dim; eps=norm_eps)
    head = Dense(dim => vocab_size)
    rope = RoPE(dim ÷ n_heads, max_len)
    SpectrogramDecoder(embed, embed_dropout, self_blocks, cross_blocks, norm, head, rope, Float32(self_attn_residual_scale))
end

function (dec::SpectrogramDecoder)(decoder_input_ids::AbstractArray{<:Integer,2}, memory::AbstractArray{T,3}) where T
    h = dec.embed(decoder_input_ids)
    h = dec.embed_dropout(h)
    seq_len = size(h, 2)
    rope_dec = dec.rope[1:seq_len]
    # Cross-attn: RoPE on q only. Use identity for krope so Zygote does not accum two
    # different-length RoPE gradients (dec vs enc) in Onion's Attention pullback.
    n_self = length(dec.self_blocks)
    scale = dec.self_attn_residual_scale
    for i in 1:n_self
        h_prev = h
        h = dec.self_blocks[i](h; rope=rope_dec, krope=rope_dec, causal=true)
        h = h_prev .+ scale .* (h .- h_prev)
        h = dec.cross_blocks[i](h, memory, memory; rope=rope_dec, krope=identity)
    end
    # Extra cross-only layers (n_cross_layers > n_decoder_layers)
    for i in (n_self + 1):length(dec.cross_blocks)
        h = dec.cross_blocks[i](h, memory, memory; rope=rope_dec, krope=identity)
    end
    dec.head(dec.norm(h))
end

# ─── Full model ───────────────────────────────────────────────────────────────

"""
    SpectrogramEncoderDecoder(encoder, decoder, ctc_head; encoder_proj=nothing)

Full model: encoder → two sibling heads (CTC and decoder), not in sequence.

  spec → Encoder → enc_mem
            ├→ ctc_head(enc_mem)     → CTC logits
            └→ [encoder_proj] → dec_mem → Decoder(ids, dec_mem) → decoder logits

CTC and decoder both consume the same encoder output; they are trained jointly
but produce different hypotheses (frame-level vs autoregressive).
"""
struct SpectrogramEncoderDecoder{E,D,C,P}
    encoder::E
    decoder::D
    ctc_head::C      # encoder output → CTC logits (CTC_VOCAB_SIZE)
    encoder_proj::P  # encoder_dim => decoder_dim (or nothing when same)
end

Flux.@layer SpectrogramEncoderDecoder

SpectrogramEncoderDecoder(encoder, decoder, ctc_head) =
    SpectrogramEncoderDecoder(encoder, decoder, ctc_head, nothing)

function SpectrogramEncoderDecoder(encoder::SpectrogramEncoder, decoder::SpectrogramDecoder)
    dim = size(decoder.embed.weight, 1)  # embedding dimension (Flux Embedding weight is (dim, vocab_size))
    SpectrogramEncoderDecoder(encoder, decoder, Dense(dim => CTC_VOCAB_SIZE), nothing)
end

"""Dispatch: no projection (same dim) vs Dense projection."""
project_memory(::Nothing, x) = x
project_memory(p::Dense, x) = p(x)

"""Return (enc_mem, dec_mem): raw encoder output (for CTC) and projected memory for decoder."""
function encode(m::SpectrogramEncoderDecoder, spec)
    enc_mem = m.encoder(spec)
    (enc_mem, project_memory(m.encoder_proj, enc_mem))
end

function (m::SpectrogramEncoderDecoder)(spec, decoder_input_ids)
    logits = m.decoder(decoder_input_ids, last(encode(m, spec)))
    return logits
end

function (m::SpectrogramEncoderDecoder)(spec, decoder_input_ids, memory)
    logits = m.decoder(decoder_input_ids, memory)
    return logits
end
