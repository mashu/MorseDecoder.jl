"""
Realistic amateur radio callsign generator.

Generates callsigns following ITU allocation patterns for major DXCC entities.
Each prefix pool is weighted roughly by contest activity levels.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  Prefix pools by region / DXCC entity
# ═══════════════════════════════════════════════════════════════════════════════

"""
Callsign prefix definition: prefix string, district digit range, suffix length range.
"""
struct PrefixDef
    prefix::String
    digits::UnitRange{Int}
    suffix_len::UnitRange{Int}
    cq_zone::Int
    weight::Float32        # relative contest activity weighting
end

const PREFIX_POOL = PrefixDef[
    # USA  (high activity)
    PrefixDef("W",  1:9, 1:3, 5,  10.0),
    PrefixDef("K",  1:9, 1:3, 5,  10.0),
    PrefixDef("N",  1:9, 1:3, 5,  8.0),
    PrefixDef("AA", 1:9, 1:2, 5,  3.0),
    PrefixDef("AB", 1:9, 1:2, 5,  2.0),
    PrefixDef("KD", 1:9, 1:3, 5,  2.0),
    PrefixDef("WA", 1:9, 1:3, 5,  2.0),
    PrefixDef("WB", 1:9, 1:3, 5,  2.0),
    # Canada
    PrefixDef("VE", 1:9, 1:3, 5,  4.0),
    PrefixDef("VA", 1:7, 1:3, 5,  2.0),
    # Germany
    PrefixDef("DL", 1:9, 1:3, 14, 5.0),
    PrefixDef("DK", 1:9, 1:3, 14, 3.0),
    PrefixDef("DJ", 1:9, 1:3, 14, 2.0),
    PrefixDef("DF", 1:9, 1:3, 14, 2.0),
    # Japan
    PrefixDef("JA", 1:9, 1:3, 25, 5.0),
    PrefixDef("JH", 1:9, 1:3, 25, 3.0),
    PrefixDef("JR", 1:9, 1:3, 25, 2.0),
    # Russia
    PrefixDef("UA", 1:9, 1:3, 16, 4.0),
    PrefixDef("RW", 1:9, 1:3, 16, 3.0),
    PrefixDef("RA", 1:9, 1:3, 16, 2.0),
    PrefixDef("RU", 1:9, 1:3, 16, 2.0),
    # UK
    PrefixDef("G",  1:9, 1:3, 14, 4.0),
    PrefixDef("M",  1:9, 1:3, 14, 2.0),
    PrefixDef("GW", 1:9, 1:3, 14, 1.0),
    # Italy
    PrefixDef("I",  1:9, 1:3, 15, 3.0),
    PrefixDef("IK", 1:9, 1:3, 15, 2.0),
    # Spain
    PrefixDef("EA", 1:9, 1:3, 14, 3.0),
    PrefixDef("EB", 1:9, 1:3, 14, 1.0),
    # France
    PrefixDef("F",  1:9, 1:3, 14, 3.0),
    # Poland
    PrefixDef("SP", 1:9, 1:3, 15, 3.0),
    PrefixDef("SQ", 1:9, 1:3, 15, 2.0),
    PrefixDef("SO", 1:9, 1:3, 15, 1.0),
    # Sweden
    PrefixDef("SM", 1:7, 1:3, 14, 2.0),
    PrefixDef("SA", 1:7, 1:3, 14, 1.0),
    # Ukraine
    PrefixDef("UR", 1:9, 1:3, 16, 2.0),
    PrefixDef("US", 1:9, 1:3, 16, 2.0),
    PrefixDef("UT", 1:9, 1:3, 16, 1.0),
    # Brazil
    PrefixDef("PY", 1:9, 1:3, 11, 3.0),
    PrefixDef("PU", 1:9, 1:3, 11, 1.0),
    # Argentina
    PrefixDef("LU", 1:9, 1:3, 13, 2.0),
    # Czech Republic
    PrefixDef("OK", 1:9, 1:3, 15, 2.0),
    PrefixDef("OL", 1:9, 1:3, 15, 1.0),
    # Netherlands
    PrefixDef("PA", 1:9, 1:3, 14, 2.0),
    PrefixDef("PD", 0:9, 1:3, 14, 1.0),
    # Finland
    PrefixDef("OH", 1:9, 1:3, 15, 2.0),
    # Australia
    PrefixDef("VK", 1:9, 1:3, 30, 2.0),
    # South Africa
    PrefixDef("ZS", 1:6, 1:3, 38, 1.5),
    # India
    PrefixDef("VU", 2:2, 1:3, 22, 1.5),
    # South Korea
    PrefixDef("HL", 1:5, 1:3, 25, 1.5),
    # China
    PrefixDef("BY", 1:9, 1:3, 24, 1.5),
    PrefixDef("BV", 1:9, 1:3, 24, 1.0),
    # Croatia
    PrefixDef("9A", 1:9, 1:3, 15, 1.5),
    # Slovenia
    PrefixDef("S5", 1:9, 1:3, 15, 1.5),
    # Caribbean / islands (contest expeditions)
    PrefixDef("PJ", 2:7, 1:2, 9,  1.0),
    PrefixDef("VP", 2:9, 1:3, 8,  0.8),
    PrefixDef("ZF", 1:2, 1:2, 8,  0.5),
    PrefixDef("V4", 7:7, 1:3, 8,  0.5),
    PrefixDef("P4", 0:0, 1:3, 9,  0.5),
]

const SUFFIX_CHARS = collect("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

"""
    random_callsign(rng) → (callsign::String, cq_zone::Int)

Generate a realistic amateur callsign using weighted prefix selection.
Returns the callsign string and its CQ zone.
"""
function random_callsign(rng::AbstractRNG)
    # Weighted selection
    weights = map(p -> p.weight, PREFIX_POOL)
    total   = sum(weights)

    r = rand(rng) * total
    cumw = Float32(0)
    chosen = first(PREFIX_POOL)
    idx = 1
    while idx ≤ length(PREFIX_POOL)
        cumw += PREFIX_POOL[idx].weight
        if r ≤ cumw
            chosen = PREFIX_POOL[idx]
            break
        end
        idx += 1
    end

    digit      = rand(rng, chosen.digits)
    suffix_len = rand(rng, chosen.suffix_len)
    suffix     = String(map(_ -> SUFFIX_CHARS[rand(rng, 1:length(SUFFIX_CHARS))],
                           1:suffix_len))

    call = "$(chosen.prefix)$(digit)$(suffix)"
    (call, chosen.cq_zone)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Operator names (common pool for Sprint / general QSOs)
# ═══════════════════════════════════════════════════════════════════════════════

const OP_NAMES = [
    "JOHN", "MIKE", "BOB", "DAVE", "JIM", "TOM", "BILL", "STEVE",
    "RICK", "DAN", "MARK", "PAUL", "GARY", "JEFF", "JACK", "FRED",
    "PETER", "RON", "KEN", "RAY", "ED", "AL", "DON", "JOE",
    "YURI", "IGOR", "HANS", "LARS", "LARS", "KARL", "STEFAN",
    "PEDRO", "JOSE", "MARCO", "TIMO", "MATTI", "OLEG", "ALEX",
    "MAREK", "ADAM", "PAWEL", "PIOTR", "KRZYSZTOF", "ANDRZEJ",
    "VLAD", "DMITRI", "SERGEI", "NIKOLAI", "TAKASHI", "HIROSHI",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  US states / Canadian provinces (for ARRL contests)
# ═══════════════════════════════════════════════════════════════════════════════

const US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
]

const CA_PROVINCES = [
    "AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE",
    "QC", "SK", "YT",
]

const ARRL_SECTIONS = [
    "CT", "EMA", "ME", "NH", "RI", "VT", "WMA",
    "ENY", "NLI", "NNJ", "SNJ", "WNY",
    "DE", "EPA", "MDC", "WPA",
    "AL", "GA", "KY", "NC", "NFL", "SC", "SFL", "WCF", "TN", "VA", "PR", "VI",
    "AR", "LA", "MS", "NM", "NTX", "OK", "STX", "WTX",
    "EB", "LAX", "ORG", "SB", "SCV", "SDG", "SF", "SJV", "SV",
    "AZ", "EWA", "ID", "MT", "NV", "OR", "UT", "WWA", "WY",
    "AK", "PAC", "HI",
    "CO", "IA", "KS", "MN", "MO", "NE", "ND", "SD",
    "IL", "IN", "WI", "MI", "OH",
    "AB", "BC", "GH", "MB", "NB", "NL", "NS", "ONE", "ONN", "ONS",
    "PE", "QC", "SK", "TER",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  Prosigns and CW abbreviations commonly heard in contests
# ═══════════════════════════════════════════════════════════════════════════════

const PROSIGNS_CQ     = ["CQ", "CQ TEST", "CQ CQ", "CQ CQ CQ", "TEST"]
const PROSIGNS_TAIL   = ["K", "KN", "BK", "AR"]
const PROSIGNS_THANKS = ["TU", "TNX", "TKS", "R", "CFM", "QSL"]
const PROSIGNS_MISC   = ["DE", "NR", "AGN", "PSE", "QRZ", "BK", "73", "GL", "UR"]
