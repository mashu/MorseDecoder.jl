"""
    vocab.jl — Token vocabulary for MorseSimulator transcript labels.

Maps simulator label string (flat_text format) to token IDs and back.
Special tokens: <START>, <END>, [S1]..[S6], [TS], [TE] (transmission start/end).
Character set: same as morse.jl (A–Z, 0–9, space, =, ?).
"""

# Special token strings as in MorseSimulator training label
const SPECIAL_START = "<START>"
const SPECIAL_END   = "<END>"
const SPECIAL_TS    = "[TS]"
const SPECIAL_TE    = "[TE]"
const SPEAKER_TAGS  = ["[S1]", "[S2]", "[S3]", "[S4]", "[S5]", "[S6]"]

# Token indices (must match model.jl)
# Chars: 1..NUM_CHARS. Then START, PAD, EOS, S1..S6, TS, TE.
const START_TOKEN_IDX = NUM_CHARS + 1
const PAD_TOKEN_IDX   = NUM_CHARS + 2
const EOS_TOKEN_IDX   = NUM_CHARS + 3   # <END>
const SPEAKER_1_IDX    = NUM_CHARS + 4
const SPEAKER_2_IDX    = NUM_CHARS + 5
const SPEAKER_3_IDX    = NUM_CHARS + 6
const SPEAKER_4_IDX    = NUM_CHARS + 7
const SPEAKER_5_IDX    = NUM_CHARS + 8
const SPEAKER_6_IDX    = NUM_CHARS + 9
const TS_TOKEN_IDX    = NUM_CHARS + 10  # [TS] transmission start
const TE_TOKEN_IDX    = NUM_CHARS + 11  # [TE] transmission end

"""Vocabulary size: chars + START + PAD + EOS + 6 speakers + TS + TE."""
const VOCAB_SIZE = NUM_CHARS + 11

speaker_token_id(k::Int) = NUM_CHARS + 3 + k
is_speaker_token(id::Int) = (SPEAKER_1_IDX <= id <= SPEAKER_6_IDX)

const _SPECIAL_TO_ID = Dict{String,Int}(
    SPECIAL_START => START_TOKEN_IDX,
    SPECIAL_END   => EOS_TOKEN_IDX,
    SPECIAL_TS    => TS_TOKEN_IDX,
    SPECIAL_TE    => TE_TOKEN_IDX,
    (SPEAKER_TAGS[i] => speaker_token_id(i) for i in 1:6)...,
)

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
        if haskey(_SPECIAL_TO_ID, seg)
            push!(ids, _SPECIAL_TO_ID[seg])
        else
            for c in seg
                if haskey(CHAR_TO_IDX, c)
                    push!(ids, CHAR_TO_IDX[c])
                end
            end
            # Space between word segments (next segment is not special)
            if i < length(parts) && !haskey(_SPECIAL_TO_ID, parts[i + 1])
                push!(ids, CHAR_TO_IDX[' '])
            end
        end
    end
    ids
end

"""
    token_ids_to_label(ids::AbstractVector{<:Integer}) -> String

Convert token IDs back to a readable label string (for logging/decode).
Stops at EOS or PAD. Renders [S1]..[S6], [TS], [TE], and characters.
"""
function token_ids_to_label(ids::AbstractVector{<:Integer})
    buf = Char[]
    for id in ids
        id == EOS_TOKEN_IDX && break
        id == PAD_TOKEN_IDX && break
        id == 0 && break
        id == START_TOKEN_IDX && continue
        if SPEAKER_1_IDX <= id <= SPEAKER_6_IDX
            append!(buf, ['[', 'S', Char('0' + (id - SPEAKER_1_IDX + 1)), ']'])
        elseif id == TS_TOKEN_IDX
            append!(buf, "[TS]")
        elseif id == TE_TOKEN_IDX
            append!(buf, "[TE]")
        elseif 1 <= id <= NUM_CHARS
            push!(buf, IDX_TO_CHAR[id])
        end
    end
    String(buf)
end

"""Plain text only (strip [S1]..[S6], [TS], [TE]) for readability."""
function token_ids_to_plain_text(ids::AbstractVector{<:Integer})
    buf = Char[]
    for id in ids
        id == EOS_TOKEN_IDX && break
        id == PAD_TOKEN_IDX && break
        id == 0 && break
        id == START_TOKEN_IDX && continue
        is_speaker_token(id) && continue
        (id == TS_TOKEN_IDX || id == TE_TOKEN_IDX) && continue
        1 <= id <= NUM_CHARS && push!(buf, IDX_TO_CHAR[id])
    end
    String(buf)
end
