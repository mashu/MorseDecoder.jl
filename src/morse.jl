"""
    morse.jl — Morse alphabet, character↔index mappings, keying envelope.

Timing follows the PARIS standard: 1 word = 50 dit units.
  • dit  = 1 unit on,  1 unit gap
  • dah  = 3 units on, 1 unit gap
  • char gap = 3 units total (1 already from last element + 2 extra)
  • word gap = 7 units
"""

# ─── Alphabet ────────────────────────────────────────────────────────────────

const MORSE_TABLE = Dict{Char,String}(
    'A' => ".-",    'B' => "-...",  'C' => "-.-.",  'D' => "-..",   'E' => ".",
    'F' => "..-.",  'G' => "--.",   'H' => "....",  'I' => "..",    'J' => ".---",
    'K' => "-.-",   'L' => ".-..",  'M' => "--",    'N' => "-.",    'O' => "---",
    'P' => ".--.",  'Q' => "--.-",  'R' => ".-.",   'S' => "...",   'T' => "-",
    'U' => "..-",   'V' => "...-",  'W' => ".--",   'X' => "-..-",  'Y' => "-.--",
    'Z' => "--..",
    '0' => "-----", '1' => ".----", '2' => "..---", '3' => "...--", '4' => "....-",
    '5' => ".....", '6' => "-....", '7' => "--...",  '8' => "---..",  '9' => "----.",
    '=' => "-...-", '?' => "..--..",
)

const ALPHABET    = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 =?"
const NUM_CHARS   = length(ALPHABET)
const CHAR_TO_IDX = Dict{Char,Int}(c => i for (i, c) in enumerate(ALPHABET))
const IDX_TO_CHAR = Dict{Int,Char}(i => c for (i, c) in enumerate(ALPHABET))

"""Encode string to index vector."""
encode_text(s::AbstractString) = [CHAR_TO_IDX[c] for c in s]

"""Decode index vector to string."""
decode_indices(v::AbstractVector{<:Integer}) = String([IDX_TO_CHAR[i] for i in v])

# ─── Timing ──────────────────────────────────────────────────────────────────

"""Nominal dit length in samples (PARIS: 60s / wpm / 50 dits)."""
dit_samples(wpm::Real, sr::Int)::Int = round(Int, 60.0 / wpm / 50.0 * sr)

"""Jittered duration — Gaussian perturbation clamped to [0.5, 2.0]× nominal."""
function jittered(nominal::Int, σ::Float32, rng::AbstractRNG)::Int
    σ == 0f0 && return nominal
    scale = clamp(1f0 + σ * randn(rng, Float32), 0.5f0, 2f0)
    round(Int, nominal * scale)
end

# ─── Keying envelope ────────────────────────────────────────────────────────

# Conservative upper bound: lead 3dit + per char at most ~25 dits (5 dahs + gaps) + trail 3dit
envelope_upper_bound(dit::Int, text_len::Int) = 6dit + max(1000, text_len * 28 * dit)

"""
    keying_envelope(text, wpm, sr, jitter, rng) → Vector{Float32}

Build the on/off keying envelope for `text`.  Returns a vector of 0s and 1s
(with jittered timing when `jitter > 0`). Single preallocated buffer, one pass.
"""
function keying_envelope(text::AbstractString, wpm::Real, sr::Int,
                         jitter::Float32, rng::AbstractRNG)
    dit = dit_samples(wpm, sr)
    j(n) = jittered(n, jitter, rng)

    cap = envelope_upper_bound(dit, length(text))
    out = Vector{Float32}(undef, cap)
    pos = 1

    # Lead-in
    len = 3dit
    @inbounds for i in pos:pos+len-1; out[i] = 0f0; end
    pos += len

    for ch in text
        if ch == ' '
            len = 7j(dit)
            @inbounds for i in pos:pos+len-1; out[i] = 0f0; end
            pos += len
            continue
        end
        morse = get(MORSE_TABLE, ch, nothing)
        isnothing(morse) && continue
        for sym in morse
            on = sym == '.' ? j(dit) : j(3dit)
            @inbounds for i in pos:pos+on-1; out[i] = 1f0; end
            pos += on
            len = j(dit)
            @inbounds for i in pos:pos+len-1; out[i] = 0f0; end
            pos += len
        end
        len = 2j(dit)
        @inbounds for i in pos:pos+len-1; out[i] = 0f0; end
        pos += len
    end

    len = 3dit
    @inbounds for i in pos:pos+len-1; out[i] = 0f0; end
    pos += len

    resize!(out, pos - 1)
    out
end
