using Test
using LinearAlgebra
using JuMP
using Clarabel
using ConstrainedCobbDouglas

@testset "Allocator: analytical and baselines" begin
    @testset "solve_unconstrained_cd_analytical: budget identity" begin
        γ = [0.5, 0.3, 0.2]
        p = [100.0, 50.0, 25.0]
        B = 10_000.0
        n, cash = solve_unconstrained_cd_analytical(γ, p, B; ε = 1e-4)
        @test sum(n .* p) + cash ≈ B atol=1e-6
        @test n[1] / n[2] ≈ (γ[1] / γ[2]) * (p[2] / p[1]) atol=1e-6
    end

    @testset "solve_unconstrained_cd_analytical: all γ ≤ 0 returns cash" begin
        γ = [-0.1, -0.05]
        p = [100.0, 50.0]
        B = 10_000.0
        n, cash = solve_unconstrained_cd_analytical(γ, p, B; ε = 1e-4)
        @test all(n .≈ 1e-4)
        @test cash ≈ B - sum(n .* p) atol=1e-6
    end

    @testset "equal_weight_target sums to 1.0" begin
        w = equal_weight_target(5)
        @test sum(w) ≈ 1.0
        @test all(w .≈ 0.2)
    end

    @testset "solve_minvar_buyhold returns weights on a feasible problem" begin
        Σ = [0.04 0.01 0.0; 0.01 0.09 0.02; 0.0 0.02 0.16]
        bounds = [0.0 1.0; 0.0 1.0; 0.0 1.0]
        w = solve_minvar_buyhold(Σ, bounds)
        @test length(w) == 3
        @test sum(w) ≈ 1.0 atol=1e-6
        @test all(w .≥ -1e-8)
    end

    @testset "solve_constrained_cd: loose-constraint identity vs analytical" begin
        γ = [0.5, 0.3, 0.2]
        p = [100.0, 50.0, 25.0]
        B = 10_000.0
        Σ = Matrix{Float64}(I, 3, 3) * 0.04
        problem = MyConstrainedCDProblem(
            γ = γ, p = p, B = B, Σ = Σ,
            σ_max = 1.0e6,            # effectively infinite
            K_turnover = 1.0e12,      # effectively infinite
            w_max = 1.0,
            n_prev = zeros(3),
            c̄ = 0.0)
        res = solve_constrained_cd(problem)
        n_an, _ = solve_unconstrained_cd_analytical(γ, p, B; ε = 1e-3)
        @test res.status == :optimal
        @test sum(res.n .* p) ≤ B + 1e-3
        # ratio test (numerical comparison, since both solve max Σ γ log n)
        for i in 1:3, j in 1:3
            if n_an[i] > 1e-6 && n_an[j] > 1e-6
                @test isapprox(res.n[i] / res.n[j], n_an[i] / n_an[j]; rtol = 5e-3)
            end
        end
    end
end
