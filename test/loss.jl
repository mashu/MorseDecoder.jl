@testset "loss" begin
    vocab = VOCAB_SIZE
    seq_len = 4
    batch = 2
    logits = randn(Float32, vocab, seq_len, batch)
    targets = rand(1:vocab, seq_len, batch)
    # Use PAD for last position in one sequence
    targets[seq_len, 2] = PAD_TOKEN_IDX

    loss = sequence_cross_entropy(logits, targets; pad_idx=PAD_TOKEN_IDX)
    @test loss isa Float32
    @test loss >= 0
    @test !isnan(loss)

    loss_smooth = sequence_cross_entropy(logits, targets; pad_idx=PAD_TOKEN_IDX, label_smoothing=0.1f0)
    @test loss_smooth >= 0
    @test !isnan(loss_smooth)
end
