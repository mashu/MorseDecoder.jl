"""
    decode.jl — Autoregressive and CTC greedy decoding.

Supports single-chunk decode, chunk-by-chunk conversation decode (streaming),
and CTC greedy decode from encoder output.
"""

using CTCLoss

# ─── GPU-friendly argmax along dim 1 ─────────────────────────────────────────

@kernel function argmax_dim1_kernel(logits, out)
    b = @index(Global)
    @uniform vocab_size = size(logits, 1)
    n_batch = size(logits, 2)
    if b <= n_batch
        best_i = 1
        best_v = @inbounds logits[1, b]
        for i in 2:vocab_size
            v = @inbounds logits[i, b]
            if v > best_v
                best_v = v
                best_i = i
            end
        end
        @inbounds out[b] = best_i
    end
end

function argmax_dim1(logits)
    backend = get_backend(logits)
    if backend isa KA.GPU
        batch_size = size(logits, 2)
        out = similar(logits, Int32, (batch_size,))
        argmax_dim1_kernel(backend, 256)(logits, out; ndrange=batch_size)
        synchronize(backend)
        reshape(Int.(out), 1, batch_size)
    else
        am = argmax(logits; dims=1)
        reshape([am[i][1] for i in eachindex(am)], 1, size(logits, 2))
    end
end

# ─── Autoregressive loop ────────────────────────────────────────────────────

"""Shared loop: run decoder from (memory, ids_buf, len_so_far) until max_len or EOS."""
function autoregressive_loop(
    model::SpectrogramEncoderDecoder,
    memory,
    ids_buf,
    len_so_far::Int;
    max_len::Int,
    to_device = identity,
)
    batch_size = size(ids_buf, 2)
    for _ in (len_so_far + 1):max_len
        ids_so_far = copy(ids_buf[1:len_so_far, :])
        logits = model.decoder(ids_so_far, memory)
        next_logits = selectdim(logits, 2, size(logits, 2))
        next_logits[PAD_TOKEN_IDX, :] .= -1f10
        next_logits[START_TOKEN_IDX, :] .= -1f10
        next_ids = argmax_dim1(next_logits)
        len_so_far += 1
        ids_buf[len_so_far, :] .= vec(next_ids)
        all(==(EOS_TOKEN_IDX), next_ids) && break
    end
    ids_buf[1:len_so_far, :]
end

# ─── Single-stream decode ───────────────────────────────────────────────────

"""
    decode_autoregressive(model, spec; max_len, start_token, to_device, batch_size)

Single-stream decode: one output sequence per spectrogram (with speaker tokens).
Stops at EOS or max_len. Returns (seq_len, batch) token indices.
"""
function decode_autoregressive(
    model::SpectrogramEncoderDecoder,
    spec;
    max_len::Int = 256,
    start_token::Int = START_TOKEN_IDX,
    to_device = identity,
    batch_size::Int = size(spec, 2),
)
    _, memory = encode(model, spec)
    ids_buf = to_device(fill(start_token, max_len, batch_size))
    autoregressive_loop(model, memory, ids_buf, 1; max_len, to_device)
end

"""
    decode_autoregressive(model, spec, initial_tokens; max_len, to_device, batch_size)

Decode continuing from prefix `initial_tokens` (for chunk-by-chunk conversation decoding).
"""
function decode_autoregressive(
    model::SpectrogramEncoderDecoder,
    spec,
    initial_tokens::AbstractVector{<:Integer};
    max_len::Int = 256,
    to_device = identity,
    batch_size::Int = 1,
)
    _, memory = encode(model, spec)
    P = length(initial_tokens)
    P >= max_len && return to_device(reshape(collect(initial_tokens), P, 1))[1:max_len, :]
    ids_buf = to_device(fill(START_TOKEN_IDX, max_len, batch_size))
    ids_buf[1:P, 1] .= vec(to_device(collect(Int, initial_tokens)))
    autoregressive_loop(model, memory, ids_buf, P; max_len, to_device)
end

# ─── Conversation decode (chunk-by-chunk) ─────────────────────────────────────

# Dispatch: (n_bins, time) → (n_bins, 1, time); already 3D passed through.
spec_for_decode(spec::AbstractMatrix) = reshape(spec, size(spec, 1), 1, size(spec, 2))
spec_for_decode(spec::AbstractArray{<:Any,3}) = spec

"""
    decode_conversation(model, chunks, to_device; max_len_per_chunk, start_token)

Decode a full conversation from an iterable of spectrogram chunks. Runs autoregressive
decode chunk by chunk; each chunk continues from the previous output.
Stops when EOS is emitted or chunks are exhausted. Returns a single token-id vector.

**Chunk source is abstract:** `chunks` can be from the simulator (`ChunkedConversation`),
from live audio (mic/radio), or from files. Each element must be a spectrogram array:
either (n_bins, time) or (n_bins, 1, time), with the same n_bins and scale as training.
"""
function decode_conversation(
    model::SpectrogramEncoderDecoder,
    chunks,
    to_device = identity;
    max_len_per_chunk::Int = 256,
    start_token::Int = START_TOKEN_IDX,
)
    tokens = [start_token]
    for chunk in chunks
        spec_3d = to_device(spec_for_decode(chunk))
        out = decode_autoregressive(
            model, spec_3d, tokens;
            max_len = max_len_per_chunk,
            to_device = to_device,
            batch_size = 1,
        )
        tokens = vec(collect(out))
        if !isempty(tokens) && tokens[end] == EOS_TOKEN_IDX
            break
        end
    end
    tokens
end

# ─── CTC greedy decode ───────────────────────────────────────────────────────

"""
    ctc_greedy_decode(model, spec; input_lengths) -> Vector{Vector{Int}}

Run encoder + CTC head, then greedy-decode: argmax per frame, collapse consecutive
duplicates, remove blanks. Returns one Vector{Int} of token IDs per batch element.
"""
function ctc_greedy_decode(
    model::SpectrogramEncoderDecoder,
    spec;
    input_lengths::Vector{Int} = fill(size(spec, 3), size(spec, 2)),
)
    enc_mem, _ = encode(model, spec)
    logits = model.ctc_head(enc_mem)
    enc_lengths = div.(input_lengths, ENCODER_DOWNSAMPLE)
    CTCLoss.ctc_greedy_decode(logits, enc_lengths; blank = CTC_BLANK_IDX)
end

"""
    ctc_greedy_decode(ctc_logits; input_lengths) -> Vector{Vector{Int}}

Greedy CTC decode from raw logits (CTC_VOCAB_SIZE, time, batch).
"""
function ctc_greedy_decode(
    ctc_logits::AbstractArray{<:Real,3};
    input_lengths::Vector{Int} = fill(size(ctc_logits, 2), size(ctc_logits, 3)),
)
    CTCLoss.ctc_greedy_decode(ctc_logits, input_lengths; blank = CTC_BLANK_IDX)
end
