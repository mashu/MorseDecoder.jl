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

"""
    prepare_ctc_targets(batch::Batch) -> Vector{Vector{Int}}

Extract CTC label sequences from a Batch. CTC is frame-level and sees only the
current chunk's spectrogram, so prefix tokens are stripped. Only the chunk's own
tokens (after prefix) are used as CTC targets.
"""
function prepare_ctc_targets(batch::Batch)
    prepare_ctc_targets(batch, div.(batch.input_lengths, ENCODER_DOWNSAMPLE))
end

function prepare_ctc_targets(batch::Batch, enc_lengths::Vector{Int})
    B = size(batch.targets, 1)
    skip = Set((START_TOKEN_IDX, PAD_TOKEN_IDX, EOS_TOKEN_IDX, 0))
    ctc_targets = Vector{Vector{Int}}(undef, B)
    for b in 1:B
        # Skip prefix, take only this chunk's tokens
        pfx = batch.prefix_lengths[b]
        tgt_end = batch.target_lengths[b]
        chunk_tgt = @view batch.targets[b, pfx+1:tgt_end]
        raw = [t for t in chunk_tgt if t ∉ skip]
        T_b = enc_lengths[b]
        L_max = max(0, div(T_b - 1, 2))   # CTC constraint: T >= 2*L+1
        ctc_targets[b] = length(raw) <= L_max ? raw : raw[1:L_max]
    end
    ctc_targets
end

# ─── CTC loss addition (dispatch: nothing = skip, Vector = compute) ──────────

"""No CTC targets → return loss unchanged."""
add_ctc_loss(loss, _model, _enc_mem, ::Nothing, _input_lengths, _ctc_weight) = loss

"""CTC targets provided → add weighted CTC loss."""
function add_ctc_loss(loss, model::SpectrogramEncoderDecoder, enc_mem,
                      ctc_targets::Vector{Vector{Int}}, input_lengths::Vector{Int},
                      ctc_weight::Real)
    logits = model.ctc_head(enc_mem)
    loss + ctc_weight * CTCLoss.ctc_loss_batched(logits, ctc_targets, input_lengths, CTC_BLANK_IDX)
end

# ─── Training step ───────────────────────────────────────────────────────────

"""
    train_step(model, spec, decoder_input, decoder_target, loss_mask; ...)

Single training step with continuation-aware loss masking.
`loss_mask` is (seq, batch) Float32 — 1.0 at positions where loss is computed.
Prefix positions have 0.0 so the model is teacher-forced on context but not
penalized for "predicting" known prefix tokens.
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
    label_smoothing::AbstractFloat = 0.0f0,
)
    enc_mem, dec_mem = encode(model, spec)
    logits = model.decoder(decoder_input, dec_mem)
    loss = sequence_cross_entropy(logits, decoder_target, loss_mask; label_smoothing)
    add_ctc_loss(loss, model, enc_mem, ctc_targets, input_lengths, ctc_weight)
end
