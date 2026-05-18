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
end
