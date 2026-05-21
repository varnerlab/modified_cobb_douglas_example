using Test
using LinearAlgebra
using Random
using Statistics
using ConstrainedCobbDouglas

@testset "SIM module" begin
    @testset "estimate_sim recovers known parameters" begin
        Random.seed!(42)
        T = 2000
        α_true, β_true, σ_ε_true = 0.03, 1.1, 0.18
        g_m = 0.08 .+ 0.30 .* randn(T)
        ε = σ_ε_true .* randn(T)
        g_i = α_true .+ β_true .* g_m .+ ε
        est = estimate_sim(g_m, g_i, "TEST")
        @test isapprox(est.α, α_true; atol = 0.02)
        @test isapprox(est.β, β_true; atol = 0.02)
        @test isapprox(est.σ_ε, σ_ε_true; atol = 0.02)
        @test est.r² > 0.5
        @test est.ticker == "TEST"
    end

    @testset "build_sim_covariance is symmetric PSD" begin
        ests = MySIMParameterEstimate[]
        for (t, β, σ_ε) in [("A", 1.0, 0.2), ("B", 0.8, 0.15), ("C", 1.3, 0.25)]
            e = MySIMParameterEstimate()
            e.ticker = t; e.α = 0.05; e.β = β; e.σ_ε = σ_ε; e.r² = 0.5
            push!(ests, e)
        end
        # σ_m and σ_ε here are in `SD of compute_market_growth output` units
        # (i.e., daily log-return × 1/Δt scale); build_sim_covariance applies
        # the Δt factor internally to return Σ in true annualized variance.
        Δt = 1.0 / 252.0
        Σ = build_sim_covariance(ests, 0.15)
        @test size(Σ) == (3, 3)
        @test Σ ≈ Σ'                      # symmetric
        @test all(eigvals(Σ) .> -1e-10)   # PSD
        @test Σ[1, 1] ≈ (1.0^2 * 0.15^2 + 0.2^2) * Δt
        @test Σ[1, 2] ≈ 1.0 * 0.8 * 0.15^2 * Δt
    end

    @testset "compute_market_growth annualizes log-returns" begin
        prices = [100.0, 101.0, 100.5, 102.0]
        g = compute_market_growth(prices; Δt = 1.0 / 252.0)
        @test length(g) == 3
        @test g[1] ≈ log(101.0 / 100.0) / (1.0 / 252.0)
    end

    @testset "compute_ema length and bounds" begin
        prices = collect(100.0:1.0:300.0)
        ema = compute_ema(prices; window = 21)
        @test length(ema) == length(prices)
        @test ema[end] > prices[1]
        @test ema[end] < prices[end]
    end

    @testset "compute_lambda is signed and tracks crossover direction" begin
        # Monotonically rising prices: short EMA ends above long EMA, so the
        # crossover ratio exceeds 1, and λ = -G·(ratio - 1) is strictly negative
        # (bullish) once both EMAs have warmed up.
        rising = collect(100.0:1.0:300.0)
        short_rising = compute_ema(rising; window = 21)
        long_rising  = compute_ema(rising; window = 63)
        λ_rising = compute_lambda(short_rising, long_rising)
        @test λ_rising[end] < 0.0
        @test all(isfinite, λ_rising)

        # Monotonically falling prices: short EMA ends below long EMA, ratio < 1,
        # λ > 0 (bearish).
        falling = collect(300.0:-1.0:100.0)
        short_falling = compute_ema(falling; window = 21)
        long_falling  = compute_ema(falling; window = 63)
        λ_falling = compute_lambda(short_falling, long_falling)
        @test λ_falling[end] > 0.0

        # Gain G scales the signal linearly.
        λ_g10 = compute_lambda(short_rising, long_rising; G = 10.0)
        @test isapprox(λ_g10[end], 10.0 * λ_rising[end]; atol = 1e-12)
    end

    @testset "compute_preference_weights returns tanh-bounded γ" begin
        sim_params = Dict("A" => (0.05, 1.0, 0.2), "B" => (0.10, 1.5, 0.3))
        tickers = ["A", "B"]
        γ = compute_preference_weights(sim_params, tickers, 0.08, 0.5)
        @test length(γ) == 2
        @test all(-1.0 .<= γ .<= 1.0)
    end

    @testset "ewls_init recovers prior estimates" begin
        s = ewls_init(0.05, 1.2, 0.18; half_life = 252.0, prior_weight = 252.0)
        @test isapprox(s.α, 0.05; atol = 1e-12)
        @test isapprox(s.β, 1.2; atol = 1e-12)
        @test isapprox(s.σ_ε, 0.18; atol = 1e-6)
        @test s.η > 0.99 && s.η < 1.0
    end

    @testset "ewls_update! tracks a step change after enough data" begin
        rng = MersenneTwister(123)
        s = ewls_init(0.0, 1.0, 0.2; half_life = 21.0, prior_weight = 21.0)
        α_new, β_new, σ_new = 0.10, 1.5, 0.20
        for _ in 1:500
            g_m = 0.15 * randn(rng)
            g_i = α_new + β_new * g_m + σ_new * randn(rng)
            ewls_update!(s, g_i, g_m)
        end
        @test isapprox(s.β, β_new; atol = 0.25)  # analytic SE ~0.24 at N=500, σ_ε=0.2, σ_m=0.15
        @test isapprox(s.α, α_new; atol = 0.10)
    end

    @testset "ewls_update! decay factor" begin
        s = ewls_init(0.0, 1.0, 0.1; half_life = 252.0, prior_weight = 252.0)
        expected_η = 2.0^(-1.0 / 252.0)
        @test isapprox(s.η, expected_η; atol = 1e-12)
    end
end
