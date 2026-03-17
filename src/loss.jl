"""
    loss.jl — Loss functions for the encoder–decoder.

Masked cross-entropy over decoder output sequences with optional label smoothing.
Supports explicit loss_mask for continuation training (prefix positions masked).
"""

"""
    sequence_cross_entropy(logits, decoder_target, loss_mask; label_smoothing)

Logits (vocab, seq, batch), decoder_target (seq, batch), loss_mask (seq, batch) Float32.
Mean CE over positions where loss_mask > 0. Prefix and pad positions are masked out.
"""
function sequence_cross_entropy(
    logits::AbstractArray{T,3},
    decoder_target::AbstractArray{<:Integer,2},
    loss_mask::AbstractArray{T,2};
    label_smoothing::T = zero(T),
) where T
    vocab, seq_len, batch = size(logits)
    log_probs = Flux.logsoftmax(logits; dims=1)
    nll_flat = -sum(Flux.onehotbatch(vec(decoder_target), 1:vocab) .* reshape(log_probs, vocab, :); dims=1)
    nll = reshape(nll_flat, seq_len, batch)
    if label_smoothing > zero(T)
        mean_log_prob = reshape(sum(log_probs; dims=1) / vocab, seq_len, batch)
        nll = (one(T) - label_smoothing) .* nll .+ label_smoothing .* (-mean_log_prob)
    end
    total_valid = max(sum(loss_mask), one(T))
    sum(nll .* loss_mask) / total_valid
end

"""
    sequence_cross_entropy(logits, decoder_target; pad_idx, label_smoothing)

Convenience method: computes loss_mask from pad_idx (all non-pad positions valid).
Use for testing or when no prefix masking is needed.
"""
function sequence_cross_entropy(
    logits::AbstractArray{T,3},
    decoder_target::AbstractArray{<:Integer,2};
    pad_idx::Int = PAD_TOKEN_IDX,
    label_smoothing::T = zero(T),
) where T
    loss_mask = T.(decoder_target .!= pad_idx)
    sequence_cross_entropy(logits, decoder_target, loss_mask; label_smoothing)
end
