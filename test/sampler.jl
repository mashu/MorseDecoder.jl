using MorseSimulator: DatasetConfig, DirectPath

function sampler_config()
    DatasetConfig(; path = DirectPath(), sample_rate = 44100, fft_size = 4096, hop_size = 128,
        n_mels = 40, f_min = 200.0, f_max = 900.0, stations = 2:4)
end

@testset "sampler" begin
    rng = MersenneTwister(123)
    cfg = sampler_config()

    s = generate_sample(cfg; rng)
    @test s isa Sample
    @test size(s.spectrogram, 1) > 0
    @test size(s.spectrogram, 2) > 0
    @test length(s.token_ids) >= 2  # at least START and END
    @test s.token_ids[1] == START_TOKEN_IDX
    @test s.token_ids[end] == EOS_TOKEN_IDX

    samples = [generate_sample(cfg; rng) for _ in 1:3]
    batch = collate(samples)
    @test batch isa Batch
    @test ndims(batch.spectrogram) == 3
    @test size(batch.spectrogram, 2) == 3
    @test size(batch.targets, 1) == 3
    @test length(batch.target_lengths) == 3
    @test length(batch.input_lengths) == 3

    # Chunked batch (small for test speed)
    batch2 = generate_training_batch(cfg, 2, 64; rng)
    @test batch2 isa Batch
    @test size(batch2.spectrogram, 2) == 2
end
