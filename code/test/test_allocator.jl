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

    @testset "solve_constrained_cd: budget deployment with c̄ > 0 and loose K_turnover" begin
        # Regression: previously the L1 turnover slacks (unbounded above,
        # absent from the objective) created degenerate directions in the
        # conic program and Clarabel terminated at a suboptimum that left
        # ~55% of the budget unspent, even when the turnover constraint
        # was provably slack. The fix detects that c̄ · max ‖n − n_prev‖_1
        # ≤ K_turnover and omits the slack encoding in that case.
        γ = [0.5, 0.3, 0.2, -0.1]
        p = [100.0, 50.0, 25.0, 200.0]
        B = 10_000.0
        Σ = Matrix{Float64}(I, 4, 4) * 0.04
        # Realistic setup: non-zero c̄, loose K_turnover, EW starting position.
        n_prev = [B / (4 * pp) for pp in p]
        problem = MyConstrainedCDProblem(
            γ = γ, p = p, B = B, Σ = Σ,
            σ_max = 1.0e6, K_turnover = 1.0e12, w_max = 1.0,
            n_prev = n_prev, c̄ = 0.05)
        res = solve_constrained_cd(problem)
        n_an, _ = solve_unconstrained_cd_analytical(γ, p, B; ε = 1e-3)

        @test res.status == :optimal
        # Budget deployment: the closed form spends 100% of B; the solver
        # must do the same to within numerical slack.
        spent_solver = sum(res.n .* p)
        @test spent_solver ≥ 0.999 * B
        @test spent_solver ≤ B + 1e-3

        # Share-vector identity on preferred subset (not just ratios).
        pref = findall(γ .> 0)
        for i in pref
            @test isapprox(res.n[i], n_an[i]; rtol = 1e-3)
        end
        # Non-preferred is pinned at ε = 1e-3 in both implementations.
        nonpref = findall(γ .<= 0)
        for i in nonpref
            @test isapprox(res.n[i], 1e-3; atol = 1e-5)
        end
    end

    @testset "solve_constrained_cd: σ_max monotonicity" begin
        γ = [0.5, 0.3, 0.2]
        p = [100.0, 50.0, 25.0]
        B = 10_000.0
        βs = [1.0, 0.8, 1.3]
        σ_m = 0.15
        σ_εs = [0.20, 0.15, 0.25]
        Σ = zeros(3, 3)
        for i in 1:3, j in 1:3
            Σ[i, j] = (i == j) ? βs[i]^2 * σ_m^2 + σ_εs[i]^2 : βs[i] * βs[j] * σ_m^2
        end
        function port_var(σ_max_val)
            problem = MyConstrainedCDProblem(
                γ = γ, p = p, B = B, Σ = Σ,
                σ_max = σ_max_val, K_turnover = 1e12, w_max = 1.0,
                n_prev = zeros(3), c̄ = 0.0)
            r = solve_constrained_cd(problem)
            return dot(r.w, Σ * r.w)
        end
        v_loose = port_var(0.50)
        v_mid   = port_var(0.20)
        v_tight = port_var(0.08)
        @test v_loose >= v_mid - 1e-6
        @test v_mid   >= v_tight - 1e-6
    end

    @testset "solve_constrained_cd: zero-turnover lock" begin
        γ = [0.5, 0.3, 0.2]
        p = [100.0, 50.0, 25.0]
        B = 10_000.0
        n_prev, _ = solve_unconstrained_cd_analytical(γ, p, B; ε = 1e-3)
        problem = MyConstrainedCDProblem(
            γ = γ, p = p, B = B, Σ = Matrix{Float64}(I, 3, 3) * 0.04,
            σ_max = 1e6, K_turnover = 0.0, w_max = 1.0,
            n_prev = n_prev, c̄ = 0.05)
        r = solve_constrained_cd(problem)
        @test r.status == :optimal
        @test all(isapprox.(r.n, n_prev; atol = 1e-4))
    end

    @testset "solve_constrained_cd: concentration cap binds" begin
        γ = [0.99, 0.005, 0.005]
        p = [100.0, 50.0, 25.0]
        B = 10_000.0
        problem = MyConstrainedCDProblem(
            γ = γ, p = p, B = B, Σ = Matrix{Float64}(I, 3, 3) * 0.04,
            σ_max = 1e6, K_turnover = 1e12, w_max = 0.40,
            n_prev = zeros(3), c̄ = 0.0)
        r = solve_constrained_cd(problem)
        @test r.status == :optimal
        @test maximum(r.w) <= 0.40 + 1e-3
    end

    @testset "solve_constrained_cd: no-preferred fallback" begin
        γ = [-0.1, -0.05, -0.2]
        p = [100.0, 50.0, 25.0]
        B = 10_000.0
        problem = MyConstrainedCDProblem(
            γ = γ, p = p, B = B, Σ = Matrix{Float64}(I, 3, 3) * 0.04,
            σ_max = 1e6, K_turnover = 1e12, w_max = 1.0,
            n_prev = zeros(3), c̄ = 0.0)
        r = solve_constrained_cd(problem)
        @test r.status == :no_preferred
        @test r.unallocated_budget > 0.0
    end

    @testset "solve_cost_aware_mv produces a feasible weight vector" begin
        γ = [0.4, 0.3, 0.2, 0.1]
        Σ = Matrix{Float64}(I, 4, 4) * 0.04
        w_prev = [0.25, 0.25, 0.25, 0.25]
        w = solve_cost_aware_mv(γ, Σ, w_prev; κ = 5.0, c = 0.001)
        @test length(w) == 4
        @test sum(w) ≈ 1.0 atol=1e-6
        @test all(w .>= -1e-8)
    end
end
