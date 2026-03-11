"""
    band.jl — Multi-station mixing: overlay N stations into one audio signal.

Simulates a busy band segment where multiple CW stations transmit
simultaneously at different carrier frequencies.  The mixed audio + station
metadata forms the basis for training data.
"""

# ─── Band scene ──────────────────────────────────────────────────────────────

"""
Result of mixing multiple stations.

Fields:
- `audio`    : mixed audio waveform (sum of all stations + noise)
- `stations` : per-station parameters
- `texts`    : per-station transmitted text (full text per station)
- `sr`       : sample rate (Hz)
- `turns`    : optional turn order `[(speaker, text), ...]` for interleaved transcript
"""
struct BandScene
    audio::Vector{Float32}
    stations::Vector{Station}
    texts::Vector{String}
    sr::Int
    turns::Union{Nothing, Vector{Tuple{Int,String}}}
end
BandScene(audio, stations, texts, sr) = BandScene(audio, stations, texts, sr, nothing)

# ─── Mixing ──────────────────────────────────────────────────────────────────

"""
    mix_stations(stations, texts, sr, rng; noise_σ) → BandScene

Synthesize each station independently, sum into a single waveform, add noise.
"""
function mix_stations(stations::AbstractVector{Station},
                      texts::AbstractVector{<:AbstractString},
                      sr::Int, rng::AbstractRNG;
                      noise_σ::Float32 = 0.02f0)
    @assert length(stations) == length(texts)

    # Generate per-station audio
    audios = [synthesize(s, t, sr, rng) for (s, t) in zip(stations, texts)]

    # Mix into common-length buffer
    max_len = maximum(length, audios)
    mixed   = zeros(Float32, max_len)
    for a in audios
        @views mixed[1:length(a)] .+= a
    end

    # Additive white Gaussian noise
    if noise_σ > 0f0
        mixed .+= noise_σ .* randn(rng, Float32, max_len)
    end

    clamp!(mixed, -1f0, 1f0)
    BandScene(mixed, collect(stations), collect(texts), sr, nothing)
end

# ─── Turn-based conversation ────────────────────────────────────────────────

"""
    mix_conversation(stations, turns, sr, rng; gap_ms, noise_σ, responder_overlap_ms) → BandScene

Mix a **conversation**: stations take turns. `turns` is a vector of (speaker_index, text)
with speaker_index in 1:length(stations). Each segment is synthesized and placed in time.

- **gap_ms**: pause between turns (can be negative for overlap).
- **responder_overlap_ms**: when the next turn is from a responder (speaker ≥ 2) and the
  previous was the runner (speaker 1), start the responder this many ms *before* the
  previous turn ends — simulates hunters calling before the runner finishes "CQ ... K".
"""
function mix_conversation(stations::AbstractVector{Station},
                          turns::AbstractVector{Tuple{Int,String}},
                          sr::Int, rng::AbstractRNG;
                          gap_ms::Real = 200f0,
                          noise_σ::Float32 = 0.02f0,
                          responder_overlap_ms::Real = 0f0)
    isempty(turns) && return BandScene(Float32[], collect(stations), ["" for _ in stations], sr, nothing)
    gap_samples = round(Int, sr * gap_ms / 1000)
    overlap_samples = max(0, round(Int, sr * responder_overlap_ms / 1000))

    # First pass: synthesize each turn
    turn_audios = [synthesize(stations[t[1]], t[2], sr, rng) for t in turns]

    # Compute total length (overlap can make next turn start earlier)
    offset = 1
    total_len = 0
    for i in 1:length(turn_audios)
        n = length(turn_audios[i])
        if i > 1
            prev_spk = turns[i - 1][1]
            spk = turns[i][1]
            if spk != 1 && prev_spk == 1 && overlap_samples > 0
                offset = max(1, offset - overlap_samples)
            else
                offset += gap_samples
            end
        end
        total_len = max(total_len, offset + n - 1)
        offset += n
    end
    mixed = zeros(Float32, total_len)

    offset = 1
    for (i, a) in enumerate(turn_audios)
        n = length(a)
        if i > 1
            prev_spk = turns[i - 1][1]
            spk = turns[i][1]
            if spk != 1 && prev_spk == 1 && overlap_samples > 0
                offset = max(1, offset - overlap_samples)
            else
                offset += gap_samples
            end
        end
        last = min(offset + n - 1, total_len)
        len_actual = last - offset + 1
        if len_actual > 0
            @views mixed[offset:last] .+= a[1:len_actual]
        end
        offset += n
    end

    if noise_σ > 0f0
        mixed .+= noise_σ .* randn(rng, Float32, total_len)
    end
    clamp!(mixed, -1f0, 1f0)

    # Full text per station (concatenate that station's segments for transcript)
    n_stations = length(stations)
    station_texts = [String[] for _ in 1:n_stations]
    for (spk, txt) in turns
        1 <= spk <= n_stations && push!(station_texts[spk], txt)
    end
    texts = [join(t, " ") for t in station_texts]

    BandScene(mixed, collect(stations), texts, sr, collect(turns))
end

"""
    random_conversation_band(rng; n_stations, sr, n_turns, kw...) → BandScene

Generate a random **turn-based** conversation: 2 or more stations, each turn is one
short message from one speaker (no overlap). Uses random_message for content.
"""
function random_conversation_band(rng::AbstractRNG;
                                  n_stations::Int = 2 + rand(rng, 0:1),  # 2 or 3
                                  sr::Int = 8000,
                                  n_turns::Int = 4 + rand(rng, 0:4),     # 4–8 turns
                                  freq_range::Tuple = (250f0, 750f0),
                                  wpm_range::Tuple = (15f0, 40f0),
                                  jitter_range::Tuple = (0.08f0, 0.25f0),
                                  amp_range::Tuple = (0.3f0, 1.0f0),
                                  noise_range::Tuple = (0.005f0, 0.08f0),
                                  gap_ms::Real = 200f0,
                                  text_fn::Function = random_text)
    n_stations = max(2, n_stations)
    freqs = spread_frequencies(rng, n_stations, freq_range)
    stations = [Station(;
        frequency = freqs[i],
        wpm      = uniform_float(rng, wpm_range...),
        jitter   = uniform_float(rng, jitter_range...),
        amplitude = uniform_float(rng, amp_range...),
    ) for i in 1:n_stations]

    # Round-robin turns: speaker 1, 2, 1, 2, ... or 1,2,3,1,2,3,...
    turns = Tuple{Int,String}[]
    for i in 1:n_turns
        speaker = mod1(i, n_stations)
        push!(turns, (speaker, text_fn(rng)))
    end

    noise_σ = uniform_float(rng, noise_range...)
    mix_conversation(stations, turns, sr, rng; gap_ms, noise_σ)
end

"""
    random_contest_conversation_band(rng; n_responders, sr, kw...) → BandScene

Contest-style: **one runner** (station 1) calling CQ and working **multiple
responders** (stations 2, 3, …) in sequence. Runner calls CQ → responder 1
sends report → runner TU → runner calls next → responder 2 report → runner TU → …
Like a real contest mult where one station works many in turn.
"""
function random_contest_conversation_band(rng::AbstractRNG;
                                         n_responders::Int = 2 + rand(rng, 0:2),  # 2–4 responders
                                         sr::Int = 8000,
                                         freq_range::Tuple = (250f0, 750f0),
                                         wpm_range::Tuple = (15f0, 40f0),
                                         jitter_range::Tuple = (0.08f0, 0.25f0),
                                         amp_range::Tuple = (0.3f0, 1.0f0),
                                         noise_range::Tuple = (0.005f0, 0.08f0),
                                         gap_ms::Real = 150f0,
                                         responder_overlap_ms::Real = 0f0)
    n_stations = 1 + max(1, n_responders)
    freqs = spread_frequencies(rng, n_stations, freq_range)
    stations = [Station(;
        frequency = freqs[i],
        wpm      = uniform_float(rng, wpm_range...),
        jitter   = uniform_float(rng, jitter_range...),
        amplitude = uniform_float(rng, amp_range...),
    ) for i in 1:n_stations]

    runner_call = random_callsign(rng)
    turns = contest_turns(rng, runner_call, n_responders)

    noise_σ = uniform_float(rng, noise_range...)
    mix_conversation(stations, turns, sr, rng; gap_ms, noise_σ, responder_overlap_ms)
end

# ─── Random band generation ─────────────────────────────────────────────────

"""
    random_band(rng; n_stations, sr, kw...) → BandScene

Generate a random multi-station band scene — the primary data source for
training.  Station frequencies are spread across the band with a minimum
separation so they remain distinguishable in the spectrogram.
"""
function random_band(rng::AbstractRNG;
                     n_stations::Int       = rand(rng, 1:4),
                     sr::Int               = 8000,
                     freq_range::Tuple     = (250f0, 750f0),
                     wpm_range::Tuple      = (15f0, 40f0),
                     jitter_range::Tuple   = (0.08f0, 0.25f0),
                     amp_range::Tuple      = (0.3f0, 1.0f0),
                     noise_range::Tuple    = (0.005f0, 0.08f0),
                     text_fn::Function     = random_text)

    freqs    = spread_frequencies(rng, n_stations, freq_range)
    stations = [Station(;
        frequency = freqs[i],
        wpm       = uniform_float(rng, wpm_range...),
        jitter    = uniform_float(rng, jitter_range...),
        amplitude = uniform_float(rng, amp_range...),
    ) for i in 1:n_stations]

    texts   = [text_fn(rng) for _ in 1:n_stations]
    noise_σ = uniform_float(rng, noise_range...)

    mix_stations(stations, texts, sr, rng; noise_σ)
end

# ─── Helpers ─────────────────────────────────────────────────────────────────

"""Sample N frequencies with minimum separation within a range."""
function spread_frequencies(rng::AbstractRNG, n::Int,
                            range::Tuple{Float32,Float32})
    lo, hi = range
    n == 0 && return Float32[]
    n == 1 && return Float32[uniform_float(rng, lo, hi)]

    # Divide band into n equal slots, pick one freq per slot
    slot = (hi - lo) / n
    freqs = Float32[lo + (i - 1) * slot + rand(rng, Float32) * slot * 0.8f0
                    for i in 1:n]
    clamp.(freqs, lo, hi)
end

"""Uniform random Float32 in [lo, hi]."""
uniform_float(rng::AbstractRNG, lo::Real, hi::Real) =
    Float32(lo) + rand(rng, Float32) * Float32(hi - lo)
