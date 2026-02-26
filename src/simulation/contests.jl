"""
Contest exchange generators.

Each contest type is a subtype of `AbstractContest`.  The `generate_exchange`
method dispatches on the contest type to produce realistic text sequences
that model what you'd actually hear on the bands.

Supported contests:
  • CQ WW DX     – RST + CQ zone (01–40)
  • CQ WPX       – RST + serial number
  • ARRL DX      – RST + state/province or RST + power
  • ARRL SS      – serial + precedence + call + check + section
  • Sprint       – serial + name + state
  • General QSO  – free-form (call, RST, name, QTH, 73)
  • IARU HF      – RST + zone or HQ multiplier

Each generator returns a `ContestExchange` with the text broken into *words*
(space-separated groups) so the simulator can apply per-word WPM drift.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  Abstract contest type
# ═══════════════════════════════════════════════════════════════════════════════

abstract type AbstractContest end

"""
Result of exchange generation: the full text string and per-word breakdown.
"""
struct ContestExchange
    text::String              # full string (spaces between words)
    words::Vector{String}     # individual "words" for per-word WPM drift
    contest_type::String      # human-readable label
end

"""Number of CW "cut numbers": 0→T, 9→N (common in contest CW)."""
const CUT_NUMBERS = Dict('0' => 'T', '9' => 'N', '1' => 'A', '5' => 'E')

"""
    maybe_cut(rng, num_str; prob=0.5) → String

Randomly replace digits with CW cut numbers used in fast contest operation.
Only applies to RST and zone/serial; controlled by probability `prob`.
"""
function maybe_cut(rng::AbstractRNG, s::AbstractString; prob::Real = 0.4)
    chars = map(collect(s)) do c
        if haskey(CUT_NUMBERS, c) && rand(rng) < prob
            CUT_NUMBERS[c]
        else
            c
        end
    end
    String(chars)
end

"""
    format_rst(rng; cut_prob=0.5) → String

Generate a signal report.  In CW contests this is almost always 5NN
(599 with cut numbers), but occasionally varies.
"""
function format_rst(rng::AbstractRNG; cut_prob::Real = 0.5)
    # 90% chance of 599, 10% chance of other RST
    if rand(rng) < 0.90
        maybe_cut(rng, "599"; prob = cut_prob)
    else
        rst = rand(rng, ["589", "579", "559", "549", "539", "599", "459"])
        maybe_cut(rng, rst; prob = cut_prob)
    end
end

"""
    format_serial(rng, max_serial; width=3) → String

Generate a serial number, sometimes with leading zeros or cut numbers.
"""
function format_serial(rng::AbstractRNG; max_serial::Int = 2000, width::Int = 3)
    n = rand(rng, 1:max_serial)
    s = Base.lpad(n, width, '0')
    maybe_cut(rng, s; prob = 0.3)
end

"""
    random_element(rng, collection)

Pick a random element from a collection.
"""
random_element(rng::AbstractRNG, v) = v[rand(rng, 1:length(v))]

# ═══════════════════════════════════════════════════════════════════════════════
#  CQ WW DX Contest
# ═══════════════════════════════════════════════════════════════════════════════

"""CQ WW DX: exchange is RST + CQ zone (01–40)."""
struct CQWorldWide <: AbstractContest end

function generate_exchange(::CQWorldWide, rng::AbstractRNG)
    my_call, my_zone  = random_callsign(rng)
    dx_call, dx_zone  = random_callsign(rng)
    rst  = format_rst(rng)
    zone = maybe_cut(rng, Base.lpad(my_zone, 2, '0'); prob = 0.3)

    # Several realistic patterns
    patterns = [
        # Running station CQ
        () -> begin
            cq = random_element(rng, PROSIGNS_CQ)
            k  = random_element(rng, PROSIGNS_TAIL)
            "$cq $my_call $my_call $k"
        end,
        # Exchange after being answered
        () -> "$dx_call $rst $zone",
        # Full QSO: CQ → answer → exchange
        () -> begin
            cq = random_element(rng, PROSIGNS_CQ)
            k  = random_element(rng, PROSIGNS_TAIL)
            "$cq $my_call $k"
        end,
        # Quick exchange in run mode
        () -> "$rst $zone $my_call",
        # Being called
        () -> "$my_call DE $dx_call $rst $zone",
        # Confirmation
        () -> begin
            tu = random_element(rng, PROSIGNS_THANKS)
            "$tu $rst $zone $my_call"
        end,
    ]

    text = random_element(rng, patterns)()
    words = split(text)
    ContestExchange(text, collect(words), "CQ WW")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  CQ WPX Contest
# ═══════════════════════════════════════════════════════════════════════════════

"""CQ WPX: exchange is RST + sequential serial number."""
struct CQWPX <: AbstractContest end

function generate_exchange(::CQWPX, rng::AbstractRNG)
    my_call, _ = random_callsign(rng)
    dx_call, _ = random_callsign(rng)
    rst    = format_rst(rng)
    serial = format_serial(rng; max_serial = 3000)

    patterns = [
        () -> begin
            cq = random_element(rng, PROSIGNS_CQ)
            k  = random_element(rng, PROSIGNS_TAIL)
            "$cq $my_call $my_call $k"
        end,
        () -> "$dx_call $rst $serial",
        () -> "$rst $serial $my_call",
        () -> begin
            tu = random_element(rng, PROSIGNS_THANKS)
            "$tu $rst $serial $my_call"
        end,
        () -> "NR $serial $serial",
        () -> "$my_call DE $dx_call $rst NR $serial",
    ]

    text = random_element(rng, patterns)()
    words = split(text)
    ContestExchange(text, collect(words), "CQ WPX")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  ARRL DX Contest
# ═══════════════════════════════════════════════════════════════════════════════

"""ARRL DX: W/VE send RST + state; DX sends RST + power."""
struct ARRLDX <: AbstractContest end

function generate_exchange(::ARRLDX, rng::AbstractRNG)
    my_call, _ = random_callsign(rng)
    dx_call, _ = random_callsign(rng)
    rst = format_rst(rng)

    # Randomly be W/VE or DX side
    if rand(rng) < 0.5
        # W/VE side: send state
        state = random_element(rng, vcat(US_STATES, CA_PROVINCES))
        patterns = [
            () -> "$dx_call $rst $state",
            () -> "$rst $state $my_call",
            () -> begin
                tu = random_element(rng, PROSIGNS_THANKS)
                "$tu $rst $state $my_call"
            end,
        ]
    else
        # DX side: send power
        power = random_element(rng, ["1KW", "KW", "100", "5W", "QRP", "K", "1K", "100W"])
        patterns = [
            () -> "$dx_call $rst $power",
            () -> "$rst $power $my_call",
            () -> begin
                tu = random_element(rng, PROSIGNS_THANKS)
                "$tu $rst $power $my_call"
            end,
        ]
    end

    text = random_element(rng, patterns)()
    words = split(text)
    ContestExchange(text, collect(words), "ARRL DX")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  ARRL Sweepstakes
# ═══════════════════════════════════════════════════════════════════════════════

"""ARRL SS: serial + precedence + callsign + check (2-digit year) + section."""
struct ARRLSweepstakes <: AbstractContest end

const SS_PRECEDENCE = ["A", "B", "M", "Q", "S", "U"]

function generate_exchange(::ARRLSweepstakes, rng::AbstractRNG)
    my_call, _ = random_callsign(rng)
    dx_call, _ = random_callsign(rng)

    serial = format_serial(rng; max_serial = 1500, width = 1)
    prec   = random_element(rng, SS_PRECEDENCE)
    check  = string(rand(rng, 50:99))   # 2-digit year first licensed
    sect   = random_element(rng, ARRL_SECTIONS)

    patterns = [
        () -> begin
            cq = random_element(rng, ["CQ SS", "CQ SWEEPSTAKES", "CQ SS CQ SS"])
            "$cq DE $my_call $my_call"
        end,
        () -> "$dx_call $serial $prec $my_call $check $sect",
        () -> "NR $serial $prec $my_call $check $sect",
        () -> begin
            tu = random_element(rng, PROSIGNS_THANKS)
            "$tu $serial $prec $my_call $check $sect"
        end,
    ]

    text = random_element(rng, patterns)()
    words = split(text)
    ContestExchange(text, collect(words), "ARRL SS")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  NA Sprint
# ═══════════════════════════════════════════════════════════════════════════════

"""Sprint: serial + name + state/province."""
struct Sprint <: AbstractContest end

function generate_exchange(::Sprint, rng::AbstractRNG)
    my_call, _ = random_callsign(rng)
    dx_call, _ = random_callsign(rng)

    serial = format_serial(rng; max_serial = 500, width = 1)
    name   = random_element(rng, OP_NAMES)
    state  = random_element(rng, US_STATES)

    patterns = [
        () -> "$dx_call $my_call $serial $name $state",
        () -> "$serial $name $state",
        () -> begin
            tu = random_element(rng, PROSIGNS_THANKS)
            "$tu $dx_call $my_call $serial $name $state"
        end,
    ]

    text = random_element(rng, patterns)()
    words = split(text)
    ContestExchange(text, collect(words), "Sprint")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  General (non-contest) QSO
# ═══════════════════════════════════════════════════════════════════════════════

"""General QSO: free-form exchange with RST, name, QTH, 73."""
struct GeneralQSO <: AbstractContest end

const QTH_POOL = [
    "STOCKHOLM", "WARSAW", "BERLIN", "LONDON", "PARIS", "TOKYO",
    "MOSCOW", "ROME", "MADRID", "PRAGUE", "VIENNA", "OSLO",
    "HELSINKI", "BRUSSELS", "AMSTERDAM", "LISBON", "DUBLIN",
    "NEW YORK", "CHICAGO", "LOS ANGELES", "TORONTO", "SAO PAULO",
    "KRAKOW", "WROCLAW", "POZNAN", "GDANSK", "LVIV", "LVOV",
    "GOETEBORG", "MALMO", "KIEV", "MINSK", "RIGA", "TALLINN",
    "BUDAPEST", "BUCHAREST", "SOFIA", "ZAGREB", "BELGRADE",
]

function generate_exchange(::GeneralQSO, rng::AbstractRNG)
    my_call, _ = random_callsign(rng)
    dx_call, _ = random_callsign(rng)
    rst  = format_rst(rng; cut_prob = 0.1)  # less cut numbers in ragchew
    name = random_element(rng, OP_NAMES)
    qth  = random_element(rng, QTH_POOL)

    patterns = [
        () -> "CQ CQ CQ DE $my_call $my_call $my_call K",
        () -> "$my_call DE $dx_call $dx_call K",
        () -> "$dx_call DE $my_call = UR RST $rst $rst = NAME $name $name = QTH $qth $qth = HW CPY? $dx_call DE $my_call KN",
        () -> "R R $dx_call DE $my_call = TNX FER CALL = UR RST $rst = NAME $name = QTH $qth = 73 $dx_call DE $my_call K",
        () -> "$dx_call DE $my_call = RST $rst = NAME IS $name = QTH $qth = BK",
        () -> "73 GL $dx_call DE $my_call SK",
        () -> "$dx_call DE $my_call = TNX QSO 73 GL = $dx_call DE $my_call SK",
    ]

    text = random_element(rng, patterns)()
    words = split(text)
    ContestExchange(text, collect(words), "General QSO")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  IARU HF Championship
# ═══════════════════════════════════════════════════════════════════════════════

"""IARU HF: RST + ITU zone or HQ station multiplier."""
struct IARUHF <: AbstractContest end

const HQ_STATIONS = [
    "ARRL", "DARC", "JARL", "RSGB", "REF", "PZK", "SSA", "SRAL",
    "SRR", "UARL", "IARU", "NRRL", "LABRE", "RAC", "OEVSV",
]

function generate_exchange(::IARUHF, rng::AbstractRNG)
    my_call, _ = random_callsign(rng)
    dx_call, _ = random_callsign(rng)
    rst = format_rst(rng)

    # Zone or HQ
    exchange = if rand(rng) < 0.85
        zone = Base.lpad(rand(rng, 1:90), 2, '0')
        maybe_cut(rng, zone; prob = 0.3)
    else
        random_element(rng, HQ_STATIONS)
    end

    patterns = [
        () -> begin
            cq = random_element(rng, ["CQ TEST", "CQ IARU", "TEST"])
            k = random_element(rng, PROSIGNS_TAIL)
            "$cq $my_call $my_call $k"
        end,
        () -> "$dx_call $rst $exchange",
        () -> "$rst $exchange $my_call",
        () -> begin
            tu = random_element(rng, PROSIGNS_THANKS)
            "$tu $rst $exchange $my_call"
        end,
    ]

    text = random_element(rng, patterns)()
    words = split(text)
    ContestExchange(text, collect(words), "IARU HF")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: pick a random contest type
# ═══════════════════════════════════════════════════════════════════════════════

const ALL_CONTESTS = AbstractContest[
    CQWorldWide(), CQWPX(), ARRLDX(), ARRLSweepstakes(),
    Sprint(), GeneralQSO(), IARUHF(),
]

"""
    random_exchange(rng) → ContestExchange

Generate a random exchange from a randomly selected contest type.
"""
function random_exchange(rng::AbstractRNG)
    contest = random_element(rng, ALL_CONTESTS)
    generate_exchange(contest, rng)
end

"""
    random_exchange(rng, contest::AbstractContest) → ContestExchange

Generate an exchange for a specific contest type.
"""
random_exchange(rng::AbstractRNG, contest::AbstractContest) =
    generate_exchange(contest, rng)
