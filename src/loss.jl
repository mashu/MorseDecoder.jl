"""
    loss.jl — Loss functions for the encoder–decoder.

Masked cross-entropy over decoder output sequences with optional label smoothing.
"""

"""
    sequence_cross_entropy(logits, decoder_target; pad_idx, label_smoothing)

Logits (vocab, seq, batch), decoder_target (seq, batch). Mean CE over non-pad positions.
When label_smoothing > 0, targets are smoothed toward uniform to reduce overconfident collapse.
"""
function sequence_cross_entropy(
    logits::AbstractArray{T,3},
    decoder_target::AbstractArray{<:Integer,2};
    pad_idx::Int = PAD_TOKEN_IDX,
    label_smoothing::T = zero(T),
) where T
    vocab, seq_len, batch = size(logits)
    log_probs = Flux.logsoftmax(logits; dims=1)
    nll_flat = -sum(Flux.onehotbatch(vec(decoder_target), 1:vocab) .* reshape(log_probs, vocab, :); dims=1)
    nll = reshape(nll_flat, seq_len, batch)
    if label_smoothing > 0
        ε = label_smoothing
        mean_log_prob = reshape(sum(log_probs; dims=1) / vocab, seq_len, batch)
        nll = (1 - ε) .* nll .+ ε .* (-mean_log_prob)
    end
    valid = decoder_target .!= pad_idx
    total_valid = max(sum(valid), 1)
    sum(nll .* valid) / total_valid
end
