"""
    vocab.jl — Token vocabulary for MorseSimulator transcript labels.

Maps simulator label string (flat_text format) to token IDs and back.
Special tokens: <START>, <END>, [S1]..[S6], [TS], [TE] (transmission start/end).
Character set: same as morse.jl (A–Z, 0–9, space, =, ?).
"""

# ─── Special token strings ──────────────────────────────────────────────────

const SPECIAL_START = "<START>"
const SPECIAL_END   = "<END>"
const SPECIAL_TS    = "[TS]"
const SPECIAL_TE    = "[TE]"
const SPEAKER_TAGS  = ["[S1]", "[S2]", "[S3]", "[S4]", "[S5]", "[S6]"]

# ─── Token indices ───────────────────────────────────────────────────────────
# Chars: 1..NUM_CHARS. Then START, PAD, EOS, S1..S6, TS, TE.

const START_TOKEN_IDX = NUM_CHARS + 1
const PAD_TOKEN_IDX   = NUM_CHARS + 2
const EOS_TOKEN_IDX   = NUM_CHARS + 3   # <END>
const SPEAKER_1_IDX   = NUM_CHARS + 4
const SPEAKER_2_IDX   = NUM_CHARS + 5
const SPEAKER_3_IDX   = NUM_CHARS + 6
const SPEAKER_4_IDX   = NUM_CHARS + 7
const SPEAKER_5_IDX   = NUM_CHARS + 8
const SPEAKER_6_IDX   = NUM_CHARS + 9
const TS_TOKEN_IDX    = NUM_CHARS + 10  # [TS] transmission start
const TE_TOKEN_IDX    = NUM_CHARS + 11  # [TE] transmission end

"""Vocabulary size: chars + START + PAD + EOS + 6 speakers + TS + TE."""
const VOCAB_SIZE = NUM_CHARS + 11

# CTC blank token is appended after VOCAB_SIZE (NNlib convention: blank = last class).
const CTC_VOCAB_SIZE = VOCAB_SIZE + 1
const CTC_BLANK_IDX  = CTC_VOCAB_SIZE

speaker_token_id(k::Int) = NUM_CHARS + 3 + k
is_speaker_token(id::Int) = (SPEAKER_1_IDX <= id <= SPEAKER_6_IDX)

"""True when `id` is a structural token ([S1]-[S6], [TS], [TE]) — not a character."""
is_structural_token(id::Int) = is_speaker_token(id) || id == TS_TOKEN_IDX || id == TE_TOKEN_IDX

const SPECIAL_TO_ID = Dict{String,Int}(
    SPECIAL_START => START_TOKEN_IDX,
    SPECIAL_END   => EOS_TOKEN_IDX,
    SPECIAL_TS    => TS_TOKEN_IDX,
    SPECIAL_TE    => TE_TOKEN_IDX,
    (SPEAKER_TAGS[i] => speaker_token_id(i) for i in 1:6)...,
)

# ─── Label → token IDs ──────────────────────────────────────────────────────

"""
    label_to_token_ids(label::String) -> Vector{Int}

Parse MorseSimulator training label into token IDs.
Label format: "<START> [TS] [S1] word word [TE] [TS] [S2] ... <END>" (space-separated).
Unknown characters are skipped.
"""
function label_to_token_ids(label::AbstractString)
    ids = Int[]
    parts = split(label)
    for (i, seg) in enumerate(parts)
        if haskey(SPECIAL_TO_ID, seg)
            push!(ids, SPECIAL_TO_ID[seg])
        else
            for c in seg
                if haskey(CHAR_TO_IDX, c)
                    push!(ids, CHAR_TO_IDX[c])
                end
            end
            # Space between word segments (next segment is not special)
            if i < length(parts) && !haskey(SPECIAL_TO_ID, parts[i + 1])
                push!(ids, CHAR_TO_IDX[' '])
            end
        end
    end
    ids
end

# ─── Token IDs → label string (dispatch-based rendering) ────────────────────

"""Append the structural-token representation for `id` to `buf`."""
function render_structural_token!(buf::Vector{Char}, id::Int)
    if is_speaker_token(id)
        append!(buf, ['[', 'S', Char('0' + (id - SPEAKER_1_IDX + 1)), ']'])
    elseif id == TS_TOKEN_IDX
        append!(buf, "[TS]")
    elseif id == TE_TOKEN_IDX
        append!(buf, "[TE]")
    end
    nothing
end

"""
    render_tokens(ids, include_structural) -> String

Shared rendering loop. When `include_structural` is true, emits speaker/TS/TE tokens;
when false, skips them (plain text only). Stops at EOS/PAD/0; skips START.
"""
function render_tokens(ids::AbstractVector{<:Integer}, include_structural::Val{V}) where V
    buf = Char[]
    for id in ids
        id == EOS_TOKEN_IDX && break
        id == PAD_TOKEN_IDX && break
        id == 0 && break
        id == START_TOKEN_IDX && continue
        if is_structural_token(id)
            V && render_structural_token!(buf, id)
            continue
        end
        1 <= id <= NUM_CHARS && push!(buf, IDX_TO_CHAR[id])
    end
    String(buf)
end

"""
    token_ids_to_label(ids) -> String

Convert token IDs back to a readable label string (for logging/decode).
Renders [S1]..[S6], [TS], [TE], and characters. Stops at EOS or PAD.
"""
token_ids_to_label(ids::AbstractVector{<:Integer}) = render_tokens(ids, Val(true))

"""Plain text only (strip [S1]..[S6], [TS], [TE]) for readability."""
token_ids_to_plain_text(ids::AbstractVector{<:Integer}) = render_tokens(ids, Val(false))
