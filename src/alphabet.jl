"""
Morse alphabet: character ↔ index mappings and dot/dash strings for the simulator.
CTC uses indices 1…NUM_CHARS for characters; index NUM_TAGS is the blank.
"""

# International Morse (A–Z, 0–9). Space has empty encoding (gap handled at word level).
const MORSE_TABLE = Dict{Char, String}(
    'A' => ".-",    'B' => "-...",  'C' => "-.-.",  'D' => "-..",   'E' => ".",
    'F' => "..-.",  'G' => "--.",   'H' => "....",  'I' => "..",   'J' => ".---",
    'K' => "-.-",   'L' => ".-..",  'M' => "--",    'N' => "-.",   'O' => "---",
    'P' => ".--.",  'Q' => "--.-",  'R' => ".-.",   'S' => "...",  'T' => "-",
    'U' => "..-",   'V' => "...-", 'W' => ".--",   'X' => "-..-", 'Y' => "-.--",
    'Z' => "--..",
    '0' => "-----", '1' => ".----", '2' => "..---", '3' => "...--", '4' => "....-",
    '5' => ".....", '6' => "-....", '7' => "--...", '8' => "---..", '9' => "----.",
    ' ' => "",      # word space; gap applied at word boundary in drift.jl
    '=' => "-...-", # BT (break); used in General QSO etc.
    '?' => "..--..", # question mark (e.g. "CPY?")
)

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 =?"
const NUM_CHARS = length(ALPHABET)
const NUM_TAGS  = NUM_CHARS + 1   # +1 for CTC blank

const CHAR_TO_IDX = Dict{Char, Int}(c => i for (i, c) in enumerate(ALPHABET))
const IDX_TO_CHAR = Dict{Int, Char}(i => c for (i, c) in enumerate(ALPHABET))
