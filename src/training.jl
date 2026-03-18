"""
    training.jl — Training step, batch preparation, and CTC target extraction.

Continuation-aware: Batch.targets = [prefix ++ chunk_tokens]. The prefix is
teacher-forced (decoder sees it) but loss-masked (no gradient signal on prefix
positions). This teaches the model that chunks are continuations.

Loss mask is built on CPU as Float32, moved to device by the training loop.
"""

using CTCLoss

# ─── Decoder batch preparation ───────────────────────────────────────────────

"""
    prepare_decoder_batch(targets, prefix_lengths; start_token, pad_token)

Build decoder input, target, and loss mask for teacher forcing.

- `targets`        : (batch, max_seq) — [prefix ++ chunk_tokens], zero-padded
- `prefix_lengths` : length of prefix per sample (these positions are loss-masked)

Returns `(; decoder_input, decoder_target, loss_mask)`:
- `decoder_input`  : (max_seq, batch) — shifted right, START prepended
- `decoder_target` : (max_seq, batch) — prediction targets
- `loss_mask`      : (max_seq, batch) Float32 — 1.0 where loss is computed, 0.0 for pad and prefix
"""
function prepare_decoder_batch(
    targets::AbstractArray{<:Integer,2},
    prefix_lengths::AbstractVector{<:Integer};
    start_token::Int = START_TOKEN_IDX,
    pad_token::Int = PAD_TOKEN_IDX,
)
    B, L = size(targets)
    target_T = permutedims(targets, (2, 1))                       # (L, B)
    input_tail = target_T[1:max(1, L) - 1, :]                    # (L-1, B)
    decoder_input = vcat(fill(start_token, 1, B), input_tail)
    decoder_target = copy(target_T)
    decoder_input = ifelse.(decoder_input .== 0, pad_token, decoder_input)
    decoder_target = ifelse.(decoder_target .== 0, pad_token, decoder_target)

    # Loss mask: 1.0 where we compute loss, 0.0 for pad and prefix positions
    loss_mask = Float32.(decoder_target .!= pad_token)
    for b in 1:B
        pl = prefix_lengths[b]
        pl > 0 && (loss_mask[1:pl, b] .= 0f0)
    end

    (; decoder_input, decoder_target, loss_mask)
end

"""
    prepare_training_batch(batch) -> (spec, decoder_input, decoder_target, loss_mask)

Unpack a Batch into tensors ready for the training step.
loss_mask is Float32 on CPU — the training loop moves it to device.
"""
function prepare_training_batch(batch::Batch)
    dec = prepare_decoder_batch(batch.targets, batch.prefix_lengths)
    (batch.spectrogram, dec.decoder_input, dec.decoder_target, dec.loss_mask)
end

# ─── CTC target preparation ──────────────────────────────────────────────────

# Tokens to skip in CTC targets: control tokens + speaker tokens.
# Speaker tokens [S1]–[S6] have no acoustic correlate (CTC cannot tell which
# station number is transmitting from the audio alone), so including them wastes
# label capacity and forces truncation of actual characters.
# [TS]/[TE] are kept because they correspond to audible silence gaps.
const CTC_SKIP_TOKENS = Set((
    START_TOKEN_IDX, PAD_TOKEN_IDX, EOS_TOKEN_IDX, 0,
    SPEAKER_1_IDX, SPEAKER_2_IDX, SPEAKER_3_IDX,
    SPEAKER_4_IDX, SPEAKER_5_IDX, SPEAKER_6_IDX,
))

"""
    prepare_ctc_targets(batch::Batch) -> Vector{Vector{Int}}

Extract CTC label sequences from a Batch. CTC is frame-level and sees only the
current chunk's spectrogram, so prefix tokens are stripped. Only the chunk's own
tokens (after prefix) are used as CTC targets.

Speaker tokens are excluded (no acoustic correlate for CTC). Transmission
boundary tokens [TS]/[TE] are kept since they correspond to silence gaps.
"""
function prepare_ctc_targets(batch::Batch)
    prepare_ctc_targets(batch, div.(batch.input_lengths, ENCODER_DOWNSAMPLE))
end

function prepare_ctc_targets(batch::Batch, enc_lengths::Vector{Int})
    B = size(batch.targets, 1)
    ctc_targets = Vector{Vector{Int}}(undef, B)
    for b in 1:B
        # Skip prefix, take only this chunk's tokens
        pfx = batch.prefix_lengths[b]
        tgt_end = batch.target_lengths[b]
        chunk_tgt = @view batch.targets[b, pfx+1:tgt_end]
        raw = [t for t in chunk_tgt if t ∉ CTC_SKIP_TOKENS]
        T_b = enc_lengths[b]
        L_max = max(0, div(T_b - 1, 2))   # CTC constraint: T >= 2*L+1
        ctc_targets[b] = length(raw) <= L_max ? raw : raw[1:L_max]
    end
    ctc_targets
end

# ─── CTC loss addition (dispatch: nothing = skip, Vector = compute) ──────────

"""No CTC targets → return loss unchanged."""
add_ctc_loss(loss, _model, _enc_mem, ::Nothing, _input_lengths, _ctc_weight; loss_balance::Real = 1.0) = loss

"""
    loss_balance_scale(model, spec, decoder_input, decoder_target, loss_mask, ctc_targets, input_lengths; label_smoothing)

One forward (no gradients) to compute decoder_loss and ctc_per_frame; returns
decoder_loss / max(ctc_per_frame, 1e-8) so that scaling the CTC term by this
value makes the two losses have equal magnitude. Use for interpretable
ctc_weight / decoder_scale (e.g. in curriculum).
"""
function loss_balance_scale(
    model::SpectrogramEncoderDecoder,
    spec,
    decoder_input,
    decoder_target,
    loss_mask,
    ctc_targets::Vector{Vector{Int}},
    input_lengths::Vector{Int};
    label_smoothing::AbstractFloat = 0.0f0,
)
    enc_mem, dec_mem = encode(model, spec)
    logits = model.decoder(decoder_input, dec_mem)
    dec = sequence_cross_entropy(logits, decoder_target, loss_mask; label_smoothing)
    logits_ctc = model.ctc_head(enc_mem)
    ctc_raw = CTCLoss.ctc_loss_batched(logits_ctc, ctc_targets, input_lengths, CTC_BLANK_IDX)
    avg_frames = Float32(sum(input_lengths)) / Float32(length(input_lengths))
    ctc_pf = ctc_raw / max(avg_frames, 1.0f0)
    Float32(dec / max(ctc_pf, 1f-8))
end

"""CTC targets provided → add weighted CTC loss, normalized per-frame.

`CTCLoss.ctc_loss_batched` returns `sum(nll) / B` (per-sample average), which is
the total negative log-likelihood of each sequence divided by batch size. This can
be 10–100× larger than the decoder's per-token cross-entropy, causing the CTC
gradient to dominate the encoder and collapse to blank.

We rescale to per-frame: divide by the average number of encoder frames so the
CTC contribution is on the same scale as the decoder's per-token loss. This way
`ctc_weight=0.5` is *roughly* a 50/50 split. The two losses are not guaranteed
equal: decoder is mean CE per *token*, CTC is per *frame* (tokens and frames
differ), and raw values can vary. Use `loss_balance` to scale CTC so that
with weight 1 and 1 the two terms have equal typical magnitude."""
function add_ctc_loss(loss, model::SpectrogramEncoderDecoder, enc_mem,
                      ctc_targets::Vector{Vector{Int}}, input_lengths::Vector{Int},
                      ctc_weight::Real; loss_balance::Real = 1.0)
    logits = model.ctc_head(enc_mem)
    ctc_raw = CTCLoss.ctc_loss_batched(logits, ctc_targets, input_lengths, CTC_BLANK_IDX)
    avg_frames = Float32(sum(input_lengths)) / Float32(length(input_lengths))
    ctc_per_frame = ctc_raw / max(avg_frames, 1.0f0)
    loss + ctc_weight * Float32(loss_balance) * ctc_per_frame
end

# ─── Training step ───────────────────────────────────────────────────────────

"""
    train_step(model, spec, decoder_input, decoder_target, loss_mask; ...)

Single training step with continuation-aware loss masking.
`loss_mask` is (seq, batch) Float32 — 1.0 at positions where loss is computed.
Prefix positions have 0.0 so the model is teacher-forced on context but not
penalized for "predicting" known prefix tokens.

Optional `decoder_scale` multiplies the decoder CE loss (e.g. 0.3 in phase 1 of
a curriculum so CTC dominates; 1.0 in phase 2).

Optional `loss_balance`: scale factor for the CTC term so that (decoder_scale * CE)
and (ctc_weight * loss_balance * ctc_per_frame) have comparable magnitude when
both weights are 1. Typically set to (decoder_loss / ctc_per_frame) from a
calibration batch.
"""
function train_step(
    model::SpectrogramEncoderDecoder,
    spec,
    decoder_input,
    decoder_target,
    loss_mask;
    ctc_targets = nothing,
    input_lengths = nothing,
    ctc_weight::AbstractFloat = 0.0f0,
    decoder_scale::AbstractFloat = 1.0f0,
    label_smoothing::AbstractFloat = 0.0f0,
    loss_balance::Real = 1.0,
)
    enc_mem, dec_mem = encode(model, spec)
    logits = model.decoder(decoder_input, dec_mem)
    loss = decoder_scale * sequence_cross_entropy(logits, decoder_target, loss_mask; label_smoothing)
    add_ctc_loss(loss, model, enc_mem, ctc_targets, input_lengths, ctc_weight; loss_balance)
end
