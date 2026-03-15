@testset "decode" begin
    # Minimal model for decode tests
    n_freq = 40
    dim = 32
    n_heads = 2
    enc = SpectrogramEncoder(n_freq, dim, n_heads, 1)
    dec = SpectrogramDecoder(VOCAB_SIZE, dim, n_heads, 1)
    model = SpectrogramEncoderDecoder(enc, dec)

    batch = 1
    time = 16
    spec = randn(Float32, n_freq, batch, time)

    # Autoregressive decode (short max_len for test)
    ids = decode_autoregressive(model, spec; max_len = 8, start_token = START_TOKEN_IDX)
    @test size(ids, 1) >= 1
    @test size(ids, 2) == batch
    @test ids[1, 1] == START_TOKEN_IDX

    # CTC greedy decode
    enc_mem, _ = encode(model, spec)
    ctc_logits = model.ctc_head(enc_mem)
    ctc_ids = ctc_greedy_decode(ctc_logits)
    @test size(ctc_ids, 2) == batch
end
