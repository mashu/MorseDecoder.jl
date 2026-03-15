@testset "morse" begin
    @test length(MORSE_TABLE) >= 26 + 10 + 2  # A-Z, 0-9, =, ?
    @test MORSE_TABLE['A'] == ".-"
    @test MORSE_TABLE['E'] == "."
    @test MORSE_TABLE['S'] == "..."
    @test MORSE_TABLE['0'] == "-----"

    @test length(ALPHABET) == NUM_CHARS
    @test ALPHABET[1:5] == "ABCDE"

    @test CHAR_TO_IDX['A'] == 1
    @test IDX_TO_CHAR[1] == 'A'
    @test encode_text("AB") == [1, 2]
    @test decode_indices([1, 2, 3]) == "ABC"

    # Round-trip
    @test decode_indices(encode_text("HELLO")) == "HELLO"

    # Timing
    @test dit_samples(20, 44100) > 0
    @test envelope_upper_bound(100, 10) > 100

    # Keying envelope (no jitter for reproducibility)
    rng = MersenneTwister(42)
    env = keying_envelope("A", 20, 44100, 0f0, rng)
    @test eltype(env) == Float32
    @test length(env) >= 1
    @test all(x -> x == 0f0 || x == 1f0, env)
end
