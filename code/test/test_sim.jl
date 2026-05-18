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
        Σ = build_sim_covariance(ests, 0.15)
        @test size(Σ) == (3, 3)
        @test Σ ≈ Σ'                      # symmetric
        @test all(eigvals(Σ) .> -1e-10)   # PSD
        @test Σ[1, 1] ≈ 1.0^2 * 0.15^2 + 0.2^2
        @test Σ[1, 2] ≈ 1.0 * 0.8 * 0.15^2
    end
end
