@testset "model" begin
    n_freq = 40
    dim = 64
    n_heads = 4
    n_enc_layers = 1
    n_dec_layers = 1
    batch = 2
    time = 32  # will become 16 after encoder downsampling

    enc = SpectrogramEncoder(n_freq, dim, n_heads, n_enc_layers)
    dec = SpectrogramDecoder(VOCAB_SIZE, dim, n_heads, n_dec_layers)
    model = SpectrogramEncoderDecoder(enc, dec)

    spec = randn(Float32, n_freq, batch, time)
    enc_mem, dec_mem = encode(model, spec)
    @test size(enc_mem, 2) == time ÷ ENCODER_DOWNSAMPLE
    @test size(enc_mem, 1) == dim
    @test size(enc_mem, 3) == batch

    seq_len = 5
    decoder_ids = fill(START_TOKEN_IDX, 1, batch)
    decoder_ids = vcat(decoder_ids, rand(1:VOCAB_SIZE, seq_len - 1, batch))
    logits = model(spec, decoder_ids, dec_mem)
    @test size(logits) == (VOCAB_SIZE, seq_len, batch)
end
