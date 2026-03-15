"""
    training.jl — Training step, batch preparation, and CTC target extraction.

Combines teacher-forced decoder CE with optional CTC loss on encoder output.
"""

using CTCLoss

# ─── Decoder batch preparation ───────────────────────────────────────────────

"""
    prepare_decoder_batch(targets; start_token, pad_token)

From Batch targets (batch, max_seq) build decoder input and target for teacher forcing.
- `decoder_input`: (max_seq, batch), [START, t1, …, t_{L-1}].
- `decoder_target`: (max_seq, batch), [t1, …, t_L]. Padding (0) is mapped to pad_token.
"""
function prepare_decoder_batch(
    targets::AbstractArray{<:Integer,2};
    start_token::Int = START_TOKEN_IDX,
    pad_token::Int = PAD_TOKEN_IDX,
)
    B, L = size(targets)
    # targets (B, L) -> (L, B) for decoder
    target_T = permutedims(targets, (2, 1))   # (L, B)
    input_tail = target_T[1:max(1, L) - 1, :]  # (L-1, B)
    decoder_input = vcat(fill(start_token, 1, B), input_tail)
    decoder_target = target_T
    decoder_input = ifelse.(decoder_input .== 0, pad_token, decoder_input)
    decoder_target = ifelse.(decoder_target .== 0, pad_token, decoder_target)
    (; decoder_input, decoder_target)
end

"""
    prepare_training_batch(batch) -> (spec, decoder_input, decoder_target)
"""
function prepare_training_batch(batch::Batch)
    dec = prepare_decoder_batch(batch.targets)
    (batch.spectrogram, dec.decoder_input, dec.decoder_target)
end

# ─── CTC target preparation ──────────────────────────────────────────────────

"""
    prepare_ctc_targets(batch::Batch) -> Vector{Vector{Int}}

Extract CTC label sequences from a Batch: one target sequence per sample.
Strips START, PAD, EOS; keeps chars, speaker tokens, [TS], [TE].
Truncates so encoder length (spectrogram ÷ ENCODER_DOWNSAMPLE) >= 2*L+1 for CTC.
"""
function prepare_ctc_targets(batch::Batch)
    prepare_ctc_targets(batch, div.(batch.input_lengths, ENCODER_DOWNSAMPLE))
end

function prepare_ctc_targets(batch::Batch, enc_lengths::Vector{Int})
    B = size(batch.targets, 1)
    ctc_targets = Vector{Vector{Int}}(undef, B)
    for b in 1:B
        tgt = @view batch.targets[b, 1:batch.target_lengths[b]]
        raw = [t for t in tgt if t != START_TOKEN_IDX && t != PAD_TOKEN_IDX && t != EOS_TOKEN_IDX && t != 0]
        T_b = enc_lengths[b]
        L_max = max(0, div(T_b - 1, 2))  # CTC needs T >= 2*L+1
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
    loss + ctc_weight * CTCLoss.ctc_loss_batched(model.ctc_head(enc_mem), ctc_targets, input_lengths, CTC_BLANK_IDX)
end

# ─── Training step ───────────────────────────────────────────────────────────

"""
    train_step(model, spec, decoder_input, decoder_target; ctc_targets, input_lengths, ctc_weight, label_smoothing)

Single training step. 100% teacher forcing for decoder. When `ctc_targets` is provided,
adds CTC loss: `loss = CE + ctc_weight * CTC`. Pass `ctc_targets=nothing` to skip CTC.
"""
function train_step(
    model::SpectrogramEncoderDecoder,
    spec,
    decoder_input,
    decoder_target;
    ctc_targets = nothing,
    input_lengths = nothing,
    ctc_weight::AbstractFloat = 0.0f0,
    label_smoothing::AbstractFloat = 0.0f0,
)
    enc_mem, dec_mem = encode(model, spec)
    logits = model.decoder(decoder_input, dec_mem)
    loss = sequence_cross_entropy(logits, decoder_target; label_smoothing)
    add_ctc_loss(loss, model, enc_mem, ctc_targets, input_lengths, ctc_weight)
end

train_step(model::SpectrogramEncoderDecoder, batch::Batch; kws...) =
    train_step(model, prepare_training_batch(batch)...; kws...)
