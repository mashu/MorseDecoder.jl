"""
    messages.jl — Random callsign + exchange text generators.

Produces diverse CW contest/QSO and report-exchange text for training.
Many patterns and variants so the decoder sees a wide distribution.
"""

# ─── Callsigns ───────────────────────────────────────────────────────────────

const PREFIXES = [
    "W", "K", "N", "AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI", "AJ", "AK", "KD", "KE", "KF",
    "VE", "VA", "VO", "VY",
    "DL", "DK", "DJ", "DM", "DO", "DP",
    "JA", "JH", "JI", "JJ", "JK", "JL", "JM", "JN", "JO", "JP", "JR", "JS", "JT", "JU", "JV", "JW", "JY",
    "G", "M", "GM", "MM", "GW", "MW",
    "F", "FW", "TM",
    "SP", "SQ", "SR", "SO", "SN", "HF",
    "UA", "RA", "UA9", "RA9", "RZ", "RW",
    "I", "IK", "IW", "IM",
    "EA", "EB", "EC", "ED", "EE", "EF",
    "SM", "SA", "SK", "SG", "7S",
    "OH", "OF", "OG", "OI",
    "VK", "VL", "VM", "VN",
    "PY", "PP", "PU", "PV",
    "OK", "OL", "OM", "OZ",
    "HA", "HG", "HI", "HK",
    "LY", "LY2", "LX",
    "S5", "S5", "S5",
    "CT", "CS", "CU",
    "YB", "YC", "YD", "YE", "YF", "YG", "YH",
    "ZL", "ZM", "ZK",
    "ZS", "ZR", "ZU", "ZV", "ZW",
]

"""Generate a realistic amateur callsign (A-Z, 0-9 only; no / so alphabet stays simple)."""
function random_callsign(rng::AbstractRNG)
    pre    = rand(rng, PREFIXES)
    digit  = rand(rng, 0:9)
    suffix = String(rand(rng, 'A':'Z', rand(rng, 1:3)))
    "$pre$digit$suffix"
end

# ─── Signal reports ──────────────────────────────────────────────────────────

const CUT_MAP = Dict('0'=>'T', '9'=>'N', '1'=>'A', '5'=>'E')

"""Random RST (often 599/5NN, sometimes weaker or different formats)."""
function random_rst(rng::AbstractRNG)
    rst = rand(rng, [
        "599", "599", "599", "5NN", "5NN",
        "589", "579", "559", "549", "539", "529",
        "339", "449", "469", "479", "489",
        "229", "339", "459",
    ])
    # Sometimes use cut numbers (5NN, 5TT, etc.)
    String([get(CUT_MAP, c, c) for c in rst if rand(rng) < 0.35 || !haskey(CUT_MAP, c)])
end

"""Random serial number (1–9999), various padding styles."""
function random_serial(rng::AbstractRNG)
    n = rand(rng, 1:9999)
    rand(rng, [
        string(n),
        lpad(n, 3, '0'),
        lpad(n, 4, '0'),
        n <= 999 ? string(n) : lpad(n, 4, '0'),
    ])
end

"""Random short serial (1–99) for quick exchanges."""
function random_serial_short(rng::AbstractRNG)
    n = rand(rng, 1:99)
    rand(rng) < 0.5 ? string(n) : lpad(n, 2, '0')
end

# ─── Exchange patterns ───────────────────────────────────────────────────────

const STATES = ["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID",
    "IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT",
    "NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"]

const OP_NAMES = ["JOHN","MIKE","BOB","DAVE","JIM","TOM","BILL","STEVE",
    "RICK","DAN","MARK","PAUL","GARY","JEFF","JACK","FRED","PETER","RON",
    "YURI","IGOR","HANS","LARS","KARL","PEDRO","MARCO","TIMO","MATTI",
    "OLEG","ALEX","MAREK","ADAM","TAKASHI","VLAD","DMITRI","ANNA","MARIA",
    "LISA","SOPHIE","ELENA","KATE","JULIA","NINA"]

const QTH_POOL = ["STOCKHOLM","WARSAW","BERLIN","LONDON","PARIS","TOKYO",
    "MOSCOW","ROME","MADRID","PRAGUE","VIENNA","OSLO","HELSINKI","CHICAGO",
    "NEW YORK","KRAKOW","WROCLAW","POZNAN","GOETEBORG","MALMO","GDANSK",
    "LODZ","SZCZECIN","BUDAPEST","BRATISLAVA","ZAGREB","BUCHAREST","SOFIA",
    "ATHENS","LISBON","DUBLIN","AMSTERDAM","BRUSSELS","COPENHAGEN"]

const ZONES = 1:40   # CQ zones

"""
    random_message(rng) → (text::String, style::Symbol)

Generate a random CW message from a large set of contest/QSO and report-exchange patterns.
Returns the text and a style label for diversity.
"""
function random_message(rng::AbstractRNG)
    call1 = random_callsign(rng)
    call2 = random_callsign(rng)
    rst   = random_rst(rng)
    zone  = lpad(rand(rng, ZONES), 2, '0')
    ser   = random_serial(rng)
    ser_s = random_serial_short(rng)
    name  = rand(rng, OP_NAMES)
    state = rand(rng, STATES)
    qth   = rand(rng, QTH_POOL)

    patterns = [
        # CQ variants
        (:cq,      "CQ CQ $call1 $call1 K"),
        (:cq,      "CQ CQ CQ $call1 K"),
        (:cq,      "CQ TEST $call1 $call1"),
        (:cq,      "CQ DX $call1 $call1 K"),
        (:cq,      "CQ CONTEST $call1 K"),
        (:cq,      "CQ $call1 $call1 DE $call1 K"),
        (:cq,      "CQ CQ DE $call1 $call1 $call1 K"),
        # Report-only / minimal
        (:report,  "$rst $ser"),
        (:report,  "$rst $zone"),
        (:report,  "$rst $zone $call1"),
        (:report,  "5NN $ser"),
        (:report,  "599 $(random_serial(rng))"),
        (:report,  "R $rst"),
        (:report,  "R $rst $call1"),
        (:report,  "NR $ser"),
        (:report,  "$ser $rst"),
        (:report,  "$zone $rst"),
        (:report,  "$ser"),
        (:report,  "TU $rst"),
        (:report,  "TU $rst $call1"),
        (:report,  "R $call2 TU $rst"),
        (:report,  "73"),
        (:report,  "73 DE $call1"),
        (:report,  "73 DE $call1 SK"),
        (:report,  "BK"),
        (:report,  "K"),
        # CQ WW style — RST + zone
        (:cqww,    "$call2 $rst $zone"),
        (:cqww,    "$rst $zone $call1"),
        (:cqww,    "$call1 $call2 $rst $zone"),
        (:cqww,    "$call2 DE $call1 $rst $zone"),
        (:cqww,    "$call1 $rst $zone"),
        (:cqww,    "5NN $zone $call1"),
        # WPX / serial contests
        (:wpx,     "$call2 $rst $ser"),
        (:wpx,     "$call1 $call2 $rst $ser"),
        (:wpx,     "NR $ser $ser"),
        (:wpx,     "$rst $ser"),
        (:wpx,     "$call2 NR $ser"),
        (:wpx,     "$ser $rst $call1"),
        # ARRL Field Day / Sprint — serial + name + state/entity
        (:sprint,  "$call2 $call1 $ser $name $state"),
        (:sprint,  "$ser $name $state"),
        (:sprint,  "$call1 $call2 $ser $name $state"),
        (:sprint,  "$call2 $ser $name $state"),
        (:sprint,  "$ser $name $qth"),
        (:sprint,  "$name $state $ser"),
        (:sprint,  "$call1 $ser $name $state"),
        # Full QSO / ragchew
        (:qso,     "CQ CQ CQ DE $call1 $call1 $call1 K"),
        (:qso,     "$call1 DE $call2 K"),
        (:qso,     "$call1 DE $call2 UR RST $rst NAME $name QTH $qth K"),
        (:qso,     "$call1 DE $call2 RST $rst $rst NAME $name $name QTH $state K"),
        (:qso,     "$call2 DE $call1 GM UR $rst $rst TNX QSO 73"),
        (:qso,     "73 DE $call1 SK"),
        (:qso,     "73 GL DE $call1 SK"),
        (:qso,     "TU $call2 73 DE $call1"),
        (:qso,     "DE $call1 $call1 K"),
        (:qso,     "$call1 $call2 R $rst 73"),
        # Short confirmations / fill-ins
        (:tu,      "TU $rst $call1"),
        (:tu,      "R $call2 TU 73"),
        (:tu,      "TU $ser"),
        (:tu,      "R $rst TU"),
        (:tu,      "CFM $rst $call1"),
        (:tu,      "QSL $rst"),
        (:tu,      "OK $call2 $rst"),
        # DE / reply starts (no CQ)
        (:reply,   "DE $call1 $call1 K"),
        (:reply,   "$call2 DE $call1 $rst $zone"),
        (:reply,   "$call1 DE $call2 $rst $ser"),
        (:reply,   "DE $call1 $call2 $rst"),
    ]

    style, text = rand(rng, patterns)
    (text, style)
end

"""Convenience: just the text."""
random_text(rng::AbstractRNG) = first(random_message(rng))

# ─── Contest runner (one caller working multiple stations) ────────────────────

"""
    contest_turns(rng, runner_call, n_responders; exchange_type, ...) → Vector{Tuple{Int,String}}

Generate turn-by-turn contest exchange following real protocol: one runner
(speaker 1) calls CQ, works each responder in turn.

- Runner **confirms** hunter: often repeats hunter's call (e.g. "SP1ABC DE W1XYZ")
  and may send report back before TU.
- Hunter sends: runner_call DE resp_call + exchange (RST+zone, RST+serial, or minimal).
- Closings: TU, TU 73, 73, or 73 DE call SK. Runner may confirm what they heard first.
- Exchange types: :cqww (RST+zone), :wpx (RST+serial), :minimal (short), :mixed (random per QSO).
- Optional shorter QSOs (just R 599 / TU) so the model sees varied length.

Returns [(speaker, text), ...] with speaker 1 = runner, 2..n = responders.
"""
function contest_turns(rng::AbstractRNG, runner_call::String, n_responders::Int;
                       exchange_type::Symbol = :mixed,  # :cqww, :wpx, :minimal, :mixed
                       allow_short_qso::Bool = true)
    n_responders = max(1, n_responders)
    turns = Tuple{Int,String}[]

    # Runner calls CQ — common patterns: CQ CQ DE call K, CQ call call K, etc.
    cq = rand(rng, [
        "CQ CQ $runner_call $runner_call K",
        "CQ CQ DE $runner_call $runner_call K",
        "CQ CQ CQ DE $runner_call $runner_call K",
        "CQ $runner_call $runner_call K",
        "CQ CQ $runner_call K",
        "CQ DX $runner_call $runner_call K",
    ])
    push!(turns, (1, cq))

    for k in 1:n_responders
        responder = 1 + k
        resp_call = random_callsign(rng)
        rst = random_rst(rng)
        # Pick exchange for this QSO
        etype = exchange_type
        if etype == :mixed
            etype = rand(rng, [:cqww, :wpx, :minimal])
        end
        # Short QSO: hunter sends minimal, runner just TU
        is_short = allow_short_qso && rand(rng) < 0.25

        # Hunter: who they're calling + DE + their call + exchange
        if is_short
            push!(turns, (responder, "$runner_call DE $resp_call R $rst"))
        elseif etype == :cqww
            zone = lpad(rand(rng, ZONES), 2, '0')
            push!(turns, (responder, "$runner_call DE $resp_call $rst $zone"))
        elseif etype == :wpx
            ser = random_serial(rng)
            push!(turns, (responder, "$runner_call DE $resp_call $rst $ser"))
        else  # minimal
            push!(turns, (responder, "$runner_call DE $resp_call $rst"))
        end

        # Runner confirms hunter (repeats hunter call + optionally report) then closes with TU/73
        if is_short
            closer = rand(rng, ["TU", "R $rst TU", "TU 73"])
            push!(turns, (1, "$resp_call $closer"))
        else
            # Sometimes runner explicitly confirms: resp_call DE runner_call R rst
            if rand(rng) < 0.5
                push!(turns, (1, "$resp_call DE $runner_call R $rst"))
            end
            # Closing: TU, TU 73, 73, CFM, etc.
            runner_close = rand(rng, [
                "$resp_call TU 73",
                "$resp_call TU",
                "R $resp_call $rst TU 73",
                "CFM $resp_call $rst TU",
                "$resp_call $rst TU",
                "73 $resp_call",
            ])
            push!(turns, (1, runner_close))
        end

        # Optional: hunter sends TU back (common in real QSOs)
        if rand(rng) < 0.55
            push!(turns, (responder, rand(rng, ["TU", "TU 73", "73"])))
        end

        # Between QSOs: runner calls next station (or CQ again)
        if k < n_responders
            next_call = random_callsign(rng)
            between = rand(rng, [
                "$next_call $next_call K",
                "CQ CQ $runner_call $runner_call K",
                "$next_call DE $runner_call K",
            ])
            push!(turns, (1, between))
        end
    end

    # Optional: runner signs off at the end (73 DE call SK)
    if rand(rng) < 0.4
        push!(turns, (1, rand(rng, ["73", "73 DE $runner_call SK", "TU 73 SK"])))
    end

    turns
end
