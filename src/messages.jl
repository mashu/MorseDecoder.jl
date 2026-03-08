"""
    messages.jl — Random callsign + exchange text generators.

Produces realistic CW contest/QSO text for training data.  Kept deliberately
simple — a handful of patterns covers the distribution well enough for the
decoder to learn.
"""

# ─── Callsigns ───────────────────────────────────────────────────────────────

const PREFIXES = [
    "W", "K", "N", "AA", "KD",          # USA
    "VE", "VA",                           # Canada
    "DL", "DK", "DJ",                     # Germany
    "JA", "JH",                           # Japan
    "G", "M",                             # UK
    "F",                                  # France
    "SP", "SQ",                           # Poland
    "UA", "RA",                           # Russia
    "I", "IK",                            # Italy
    "EA",                                 # Spain
    "SM", "SA",                           # Sweden
    "OH",                                 # Finland
    "VK",                                 # Australia
    "PY",                                 # Brazil
    "OK",                                 # Czech Republic
]

"""Generate a realistic amateur callsign."""
function random_callsign(rng::AbstractRNG)
    pre    = rand(rng, PREFIXES)
    digit  = rand(rng, 1:9)
    suffix = String(rand(rng, 'A':'Z', rand(rng, 1:3)))
    "$pre$digit$suffix"
end

# ─── Signal reports ──────────────────────────────────────────────────────────

const CUT_MAP = Dict('0'=>'T', '9'=>'N', '1'=>'A', '5'=>'E')

"""Random RST (usually 599, occasionally weaker)."""
function random_rst(rng::AbstractRNG)
    rst = rand(rng) < 0.85 ? "599" : rand(rng, ["589","579","559","549","539"])
    # Maybe apply cut numbers (contest shorthand: 5NN = 599)
    String([get(CUT_MAP, c, c) for c in rst if rand(rng) < 0.4 || !haskey(CUT_MAP, c)])
end

"""Random serial number (1–2000), optionally zero-padded."""
function random_serial(rng::AbstractRNG)
    n = rand(rng, 1:2000)
    rand(rng) < 0.5 ? string(n) : lpad(n, 3, '0')
end

# ─── Exchange patterns ───────────────────────────────────────────────────────

const STATES = ["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID",
    "IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT",
    "NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"]

const OP_NAMES = ["JOHN","MIKE","BOB","DAVE","JIM","TOM","BILL","STEVE",
    "RICK","DAN","MARK","PAUL","GARY","JEFF","JACK","FRED","PETER","RON",
    "YURI","IGOR","HANS","LARS","KARL","PEDRO","MARCO","TIMO","MATTI",
    "OLEG","ALEX","MAREK","ADAM","TAKASHI","VLAD","DMITRI"]

const QTH_POOL = ["STOCKHOLM","WARSAW","BERLIN","LONDON","PARIS","TOKYO",
    "MOSCOW","ROME","MADRID","PRAGUE","VIENNA","OSLO","HELSINKI","CHICAGO",
    "NEW YORK","KRAKOW","WROCLAW","POZNAN","GOETEBORG","MALMO"]

"""
    random_message(rng) → (text::String, style::Symbol)

Generate a random CW message from one of several contest/QSO patterns.
Returns the text and a label for the exchange style.
"""
function random_message(rng::AbstractRNG)
    call1 = random_callsign(rng)
    call2 = random_callsign(rng)
    rst   = random_rst(rng)

    patterns = [
        # CQ call
        (:cq,      "CQ CQ $call1 $call1 K"),
        (:cq,      "CQ TEST $call1 $call1"),
        # CQ WW — RST + zone
        (:cqww,    "$call2 $rst $(lpad(rand(rng,1:40),2,'0'))"),
        (:cqww,    "$rst $(lpad(rand(rng,1:40),2,'0')) $call1"),
        # WPX — RST + serial
        (:wpx,     "$call2 $rst $(random_serial(rng))"),
        (:wpx,     "NR $(random_serial(rng)) $(random_serial(rng))"),
        # Sprint — serial + name + state
        (:sprint,  "$call2 $call1 $(random_serial(rng)) $(rand(rng,OP_NAMES)) $(rand(rng,STATES))"),
        (:sprint,  "$(random_serial(rng)) $(rand(rng,OP_NAMES)) $(rand(rng,STATES))"),
        # General QSO
        (:qso,     "CQ CQ CQ DE $call1 $call1 $call1 K"),
        (:qso,     "$call1 DE $call2 UR RST $rst NAME $(rand(rng,OP_NAMES)) QTH $(rand(rng,QTH_POOL)) K"),
        (:qso,     "73 DE $call1 SK"),
        # Confirmation
        (:tu,      "TU $rst $call1"),
        (:tu,      "R $call2 TU 73"),
    ]

    style, text = rand(rng, patterns)   # patterns are (Symbol, String)
    (text, style)
end

"""Convenience: just the text."""
random_text(rng::AbstractRNG) = first(random_message(rng))
