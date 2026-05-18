using Test
using LinearAlgebra
using Random
using Statistics
using JLD2
using JumpHMM
using ConstrainedCobbDouglas

@testset "MPC module" begin
    @testset "Closed-form arm: single-asset moments" begin
        α = 0.05; β = 1.0; σ_ε = 0.20; σ_m = 0.15
        T = 21; Δt = 1.0 / 252.0
        # one-asset projection: V_T closed-form
        w = [1.0]
        μ_arr, σ_arr = forward_project_closed_form(
            [α], [β], σ_m, [σ_ε], w, 100_000.0, T, Δt)
        @test length(μ_arr) == T
        # Variance per Δt = β²σ_m² + σ_ε² ; over T steps: var(log V_T/V_0) = ... · T·Δt
        var_per_step = β^2 * σ_m^2 + σ_ε^2
        var_total = var_per_step * T * Δt
        # σ_T (in dollar units) ≈ V_0 · sqrt(var_total) for small variance
        @test σ_arr[end] > 0.0
    end

    @testset "check_trigger: band exit" begin
        proj = MyMPCProjection(
            μ = [100.0, 101.0, 102.0],
            σ = [1.0, 1.0, 1.0],
            V₀ = 100.0, paths = zeros(1, 3),
            decision_date_idx = 1,
            closed_form_μ = [100.0, 101.0, 102.0],
            closed_form_σ = [1.0, 1.0, 1.0],
            divergence_warning = false)
        spec = MyMPCSpec(z = 1.96, T = 21, N = 10, D_max = 0.20)
        state = MyBacktestState()
        state.date_idx = 3; state.last_decision_t = 1
        state.V_t = 110.0    # well above μ + zσ at τ=2
        state.wealth_peak = 110.0
        state.last_projection = proj
        trig = check_trigger(state, spec)
        @test trig.fired == true
        @test trig.reason == :band_exit
    end

    @testset "check_trigger: horizon elapsed" begin
        proj = MyMPCProjection(
            μ = fill(100.0, 21), σ = fill(1.0, 21),
            V₀ = 100.0, paths = zeros(1, 21),
            decision_date_idx = 1,
            closed_form_μ = fill(100.0, 21),
            closed_form_σ = fill(1.0, 21),
            divergence_warning = false)
        spec = MyMPCSpec(z = 1.96, T = 21, N = 10, D_max = 0.20)
        state = MyBacktestState()
        state.date_idx = 22; state.last_decision_t = 1   # τ = 21
        state.V_t = 100.0; state.wealth_peak = 100.0
        state.last_projection = proj
        trig = check_trigger(state, spec)
        @test trig.fired == true
        @test trig.reason == :horizon_elapsed
    end

    @testset "check_trigger: drawdown circuit breaker" begin
        proj = MyMPCProjection(
            μ = fill(100.0, 21), σ = fill(50.0, 21),   # very wide band
            V₀ = 100.0, paths = zeros(1, 21),
            decision_date_idx = 1,
            closed_form_μ = fill(100.0, 21),
            closed_form_σ = fill(50.0, 21),
            divergence_warning = false)
        spec = MyMPCSpec(z = 1.96, T = 21, N = 10, D_max = 0.08)
        state = MyBacktestState()
        state.date_idx = 2; state.last_decision_t = 1
        state.V_t = 90.0; state.wealth_peak = 100.0    # 10% drawdown
        state.last_projection = proj
        trig = check_trigger(state, spec)
        @test trig.fired == true
        @test trig.reason == :drawdown
    end

    @testset "check_trigger: in-spec idle" begin
        proj = MyMPCProjection(
            μ = fill(100.0, 21), σ = fill(2.0, 21),
            V₀ = 100.0, paths = zeros(1, 21),
            decision_date_idx = 1,
            closed_form_μ = fill(100.0, 21),
            closed_form_σ = fill(2.0, 21),
            divergence_warning = false)
        spec = MyMPCSpec(z = 1.96, T = 21, N = 10, D_max = 0.20)
        state = MyBacktestState()
        state.date_idx = 3; state.last_decision_t = 1
        state.V_t = 100.5; state.wealth_peak = 100.5
        state.last_projection = proj
        trig = check_trigger(state, spec)
        @test trig.fired == false
        @test trig.reason == :in_spec
    end
end
