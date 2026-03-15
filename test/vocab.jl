@testset "vocab" begin
    @test START_TOKEN_IDX == NUM_CHARS + 1
    @test PAD_TOKEN_IDX == NUM_CHARS + 2
    @test EOS_TOKEN_IDX == NUM_CHARS + 3
    @test SPEAKER_1_IDX == NUM_CHARS + 4
    @test TS_TOKEN_IDX == NUM_CHARS + 10
    @test TE_TOKEN_IDX == NUM_CHARS + 11
    @test VOCAB_SIZE == NUM_CHARS + 11
    @test CTC_VOCAB_SIZE == VOCAB_SIZE + 1
    @test CTC_BLANK_IDX == CTC_VOCAB_SIZE

    @test speaker_token_id(1) == SPEAKER_1_IDX
    @test is_speaker_token(SPEAKER_1_IDX)
    @test !is_speaker_token(1)

    # Label round-trip
    label = "<START> [TS] [S1] HELLO [TE] [TS] [S2] WORLD [TE] <END>"
    ids = label_to_token_ids(label)
    @test ids[1] == START_TOKEN_IDX
    @test ids[end] == EOS_TOKEN_IDX
    @test length(ids) > 5

    back = token_ids_to_label(ids)
    @test occursin("HELLO", back)
    @test occursin("WORLD", back)

    plain = token_ids_to_plain_text(ids)
    @test occursin("HELLO", plain)
    @test occursin("WORLD", plain)
end
