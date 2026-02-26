"""
WAV I/O — minimal 16-bit PCM mono, no external dependency.

Part of the audio submodule: read/write RIFF/WAVE for inspection and dataset storage.
"""

"""
    write_wav(path, audio, sample_rate)

Write a `Vector{Float32}` as a 16-bit PCM WAV file.
"""
function write_wav(path::AbstractString, audio::AbstractVector{<:AbstractFloat},
                   sample_rate::Int)
    samples = round.(Int16, clamp.(audio, -1f0, 1f0) .* Int16(32767))
    n = length(samples)
    data_size = n * 2
    file_size = 36 + data_size

    open(path, "w") do io
        write(io, b"RIFF")
        write(io, UInt32(file_size))
        write(io, b"WAVE")
        write(io, b"fmt ")
        write(io, UInt32(16))
        write(io, UInt16(1))
        write(io, UInt16(1))
        write(io, UInt32(sample_rate))
        write(io, UInt32(sample_rate * 2))
        write(io, UInt16(2))
        write(io, UInt16(16))
        write(io, b"data")
        write(io, UInt32(data_size))
        write(io, samples)
    end
    path
end

"""
    read_wav(path) → (audio::Vector{Float32}, sample_rate::Int)

Read a 16-bit PCM mono WAV file. Returns normalised Float32 audio.
"""
function read_wav(path::AbstractString)
    open(path, "r") do io
        riff = read(io, 4)
        riff == b"RIFF" || error("Not a RIFF file: $path")
        read(io, UInt32)
        wave = read(io, 4)
        wave == b"WAVE" || error("Not a WAVE file: $path")
        sample_rate = 0
        audio = Float32[]
        while !eof(io)
            chunk_id   = read(io, 4)
            chunk_size = read(io, UInt32)
            if chunk_id == b"fmt "
                format = read(io, UInt16)
                read(io, UInt16)
                sample_rate = Int(read(io, UInt32))
                skip(io, chunk_size - 8)
            elseif chunk_id == b"data"
                n_samples = Int(chunk_size) ÷ 2
                raw = Vector{Int16}(undef, n_samples)
                read!(io, raw)
                audio = Float32.(raw) ./ 32767f0
            else
                skip(io, chunk_size)
            end
        end
        (audio, sample_rate)
    end
end
