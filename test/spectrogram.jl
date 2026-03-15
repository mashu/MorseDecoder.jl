@testset "spectrogram" begin
    cfg = SpectrogramConfig()
    @test cfg.nfft == 1024
    @test cfg.hop == 64
    @test cfg.freq_lo == 200f0
    @test cfg.freq_hi == 800f0

    sr = 8000
    @test num_bins(cfg, sr) > 0
    @test num_frames(cfg, 16000) >= 1

    # spectrogram_to_model_scale: log10 and clip
    spec = Float32.(rand(10, 20))
    scaled = spectrogram_to_model_scale(spec)
    @test size(scaled) == size(spec)
    @test eltype(scaled) == Float32
end
