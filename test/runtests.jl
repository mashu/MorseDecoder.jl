using MorseDecoder
using Random
using Test

@testset "MorseDecoder.jl" begin
    include("morse.jl")
    include("vocab.jl")
    include("spectrogram.jl")
    include("model.jl")
    include("loss.jl")
    include("sampler.jl")
    include("decode.jl")
end
