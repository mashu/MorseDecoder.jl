@testset "spectrogram" begin
    cfg = SpectrogramConfig()
    @test cfg.nfft == 4096
    @test cfg.hop == 128
    @test cfg.freq_lo == 200f0
    @test cfg.freq_hi == 900f0

    sr = 44100
    @test num_bins(cfg, sr) > 0
    @test num_frames(cfg, 16000) >= 1

    # spectrogram_to_model_scale: log10 and clip
    spec = Float32.(rand(10, 20))
    scaled = spectrogram_to_model_scale(spec)
    @test size(scaled) == size(spec)
    @test eltype(scaled) == Float32
end
