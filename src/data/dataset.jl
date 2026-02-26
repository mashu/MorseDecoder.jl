"""
Dataset generation, storage, and loading.

Manifest (CSV), dataset generation (WAV + manifest), and lazy loading for training.
WAV I/O is in audio/wav.jl; spectrogram in audio/spectrogram.jl.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  Manifest (CSV-based)
# ═══════════════════════════════════════════════════════════════════════════════

struct ManifestEntry
    file_id::String
    text::String
    contest_type::String
    wpm::Float32
    pitch::Float32
    noise_power::Float32
    amplitude::Float32
    max_drift::Float32
    duration_s::Float32
    n_audio_samples::Int
    n_spec_frames::Int
end

"""
    write_manifest(path, entries)

Write a Vector{ManifestEntry} as a CSV file.
"""
function write_manifest(path::AbstractString, entries::AbstractVector{ManifestEntry})
    open(path, "w") do io
        println(io, "file_id,text,contest_type,wpm,pitch,noise_power,amplitude," *
                    "max_drift,duration_s,n_audio_samples,n_spec_frames")
        map(entries) do e
            escaped = replace(replace(e.text, "\"" => "\"\""), "\n" => " ")
            println(io, "$(e.file_id),\"$(escaped)\",$(e.contest_type)," *
                       "$(e.wpm),$(e.pitch),$(e.noise_power),$(e.amplitude)," *
                       "$(e.max_drift),$(e.duration_s),$(e.n_audio_samples)," *
                       "$(e.n_spec_frames)")
            nothing
        end
    end
    path
end

"""
    read_manifest(path) → Vector{ManifestEntry}
"""
function read_manifest(path::AbstractString)
    lines = readlines(path)
    length(lines) ≤ 1 && return ManifestEntry[]

    map(lines[2:end]) do line
        parts = parse_csv_line(line)
        ManifestEntry(
            parts[1], parts[2], parts[3],
            parse(Float32, parts[4]), parse(Float32, parts[5]),
            parse(Float32, parts[6]), parse(Float32, parts[7]),
            parse(Float32, parts[8]), parse(Float32, parts[9]),
            parse(Int, parts[10]), parse(Int, parts[11]),
        )
    end
end

"""Simple CSV line parser that handles one quoted field (the text column)."""
function parse_csv_line(line::AbstractString)
    parts = String[]
    i = 1
    n = length(line)
    while i ≤ n
        if line[i] == '"'
            j = i + 1
            while j ≤ n
                if line[j] == '"'
                    if j < n && line[j + 1] == '"'
                        j += 2
                    else
                        break
                    end
                else
                    j += 1
                end
            end
            field = replace(line[i+1:j-1], "\"\"" => "\"")
            push!(parts, field)
            i = j + 2
        else
            j = findnext(',', line, i)
            if isnothing(j)
                push!(parts, line[i:end])
                break
            else
                push!(parts, line[i:j-1])
                i = j + 1
            end
        end
    end
    parts
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset generation
# ═══════════════════════════════════════════════════════════════════════════════

Base.@kwdef struct DatasetConfig
    n_samples::Int          = 10_000
    output_dir::String      = "dataset"
    sample_rate::Int        = 2000
    pitch_range::UnitRange{Int}  = 300:900
    wpm_range::UnitRange{Int}    = 18:45
    noise_range::UnitRange{Int}  = 0:200
    amp_range::UnitRange{Int}    = 20:150
    max_drift::Float32      = 5f0
    contest::Union{AbstractContest, Nothing} = nothing
    stratify_contests::Bool = false
end

"""
    generate_dataset(config; rng) → Vector{ManifestEntry}
"""
function generate_dataset(config::DatasetConfig;
                          rng::AbstractRNG = Random.default_rng())

    wav_dir = joinpath(config.output_dir, "wav")
    mkpath(wav_dir)

    contest_for_sample(i) = if config.stratify_contests
        ALL_CONTESTS[mod1(i, length(ALL_CONTESTS))]
    else
        config.contest
    end

    entries = map(1:config.n_samples) do i
        file_id = Base.lpad(i, 6, '0')

        sig, exch = generate_contest_signal(;
            rng,
            sample_rate = config.sample_rate,
            pitch_range = config.pitch_range,
            wpm_range   = config.wpm_range,
            noise_range = config.noise_range,
            amp_range   = config.amp_range,
            max_drift   = config.max_drift,
            contest     = contest_for_sample(i),
        )

        wav_path = joinpath(wav_dir, "$(file_id).wav")
        write_wav(wav_path, sig.audio, config.sample_rate)

        pitch_hz = carrier_frequency(sig.config.modulation)
        noise_pw = noise_power(sig.config.noise)
        drift    = max_drift(sig.config.keying)

        duration_s = length(sig.audio) / config.sample_rate

        if i % 1000 == 0
            println("  Generated $i / $(config.n_samples) " *
                    "($(exch.contest_type): \"$(first(exch.text, 40))...\")")
        end

        ManifestEntry(file_id, sig.text, exch.contest_type,
                      sig.config.wpm, pitch_hz, noise_pw,
                      sig.config.amplitude, drift,
                      Float32(duration_s), length(sig.audio),
                      size(sig.spectrogram, 2))
    end

    manifest_path = joinpath(config.output_dir, "manifest.csv")
    write_manifest(manifest_path, entries)

    println("\n✓ Dataset generated:")
    println("  $(config.n_samples) samples in $(config.output_dir)/wav/")
    println("  Manifest: $(manifest_path)")
    total_bytes = sum(e -> e.n_audio_samples * 2 + 44, entries)
    println("  Total WAV size: $(round(total_bytes / 1e6; digits=1)) MB")

    entries
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset loading for training
# ═══════════════════════════════════════════════════════════════════════════════

struct DatasetLoader
    manifest::Vector{ManifestEntry}
    wav_dir::String
    sample_rate::Int
    chunk_duration_s::Float64
    nfft::Int
end

"""Number of frequency bins per token (100–900 Hz band)."""
num_freq_bins(loader::DatasetLoader) =
    n_bins(MorseFFTConfig(loader.sample_rate, loader.chunk_duration_s, loader.nfft))

function load_dataset(dir::AbstractString; sample_rate::Int = 2000,
                      chunk_duration_s::Real = 0.04,
                      nfft::Int = 512)
    manifest = read_manifest(joinpath(dir, "manifest.csv"))
    DatasetLoader(manifest, joinpath(dir, "wav"), sample_rate, Float64(chunk_duration_s), nfft)
end

Base.length(d::DatasetLoader) = length(d.manifest)

"""Return (audio at target_sr, target_sr)."""
resample_to(audio::AbstractVector, sr::Int, target_sr::Int) =
    sr == target_sr ? (audio, sr) : (Float32.(DSP.resample(audio, target_sr / sr)), target_sr)

function load_sample(loader::DatasetLoader, i::Int)
    entry = loader.manifest[i]
    audio, sr = resample_to(read_wav(joinpath(loader.wav_dir, "$(entry.file_id).wav"))..., loader.sample_rate)
    spec = audio_to_tokens(audio, MorseFFTConfig(sr, loader.chunk_duration_s, loader.nfft))
    tgt = map(c -> CHAR_TO_IDX[c], collect(entry.text))
    (spec, tgt, size(spec, 2), length(tgt))
end

"""
    collate_batch(samples) → (specs, targets, input_lengths, target_lengths)

Pad a vector of (spec, target, in_len, tgt_len) tuples to uniform size.
"""
function collate_batch(samples::AbstractVector)
    B = length(samples)
    specs, tgts, in_lens, tgt_lens = ntuple(i -> map(s -> s[i], samples), 4)
    freq_bins = size(first(specs), 1)
    max_time = maximum(in_lens)
    max_tgt_len = maximum(tgt_lens)
    spec_batch = zeros(Float32, freq_bins, B, max_time)
    tgt_batch = zeros(Int, B, max_tgt_len)
    for i in 1:B
        spec_batch[:, i, 1:in_lens[i]] .= specs[i]
        tgt_batch[i, 1:tgt_lens[i]] .= tgts[i]
    end
    (spec_batch, tgt_batch, Vector{Int}(in_lens), Vector{Int}(tgt_lens))
end

function load_batch(loader::DatasetLoader, indices::AbstractVector{Int})
    samples = map(i -> load_sample(loader, i), indices)
    collate_batch(samples)
end

function random_training_batch(loader::DatasetLoader, batch_size::Int;
                               rng::AbstractRNG = Random.default_rng())
    indices = rand(rng, 1:length(loader), batch_size)
    load_batch(loader, indices)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset statistics
# ═══════════════════════════════════════════════════════════════════════════════

function dataset_stats(entries::AbstractVector{ManifestEntry})
    n = length(entries)
    durations = map(e -> e.duration_s, entries)
    wpms      = map(e -> e.wpm, entries)
    pitches   = map(e -> e.pitch, entries)
    noises    = map(e -> e.noise_power, entries)
    contests  = map(e -> e.contest_type, entries)
    contest_counts = Dict{String,Int}()
    map(contests) do c
        contest_counts[c] = get(contest_counts, c, 0) + 1
        nothing
    end
    total_audio_s = sum(durations)
    total_wav_mb  = sum(e -> (e.n_audio_samples * 2 + 44), entries) / 1e6
    (
        n_samples     = n,
        total_audio_h = total_audio_s / 3600,
        total_wav_mb  = total_wav_mb,
        duration_mean = Statistics.mean(durations),
        duration_std  = Statistics.std(durations),
        wpm_range     = (minimum(wpms), maximum(wpms)),
        pitch_range   = (minimum(pitches), maximum(pitches)),
        noise_range   = (minimum(noises), maximum(noises)),
        contest_dist  = contest_counts,
    )
end
