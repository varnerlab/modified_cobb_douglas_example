# --- Analytical unconstrained Cobb-Douglas (vendored from Compute.jl
# allocate_cobb_douglas, lines 505-541). Used as strategy 3 and strategy 5
# allocator and as the loose-constraint reference for the JuMP solver. ---

"""
    solve_unconstrained_cd_analytical(γ, p, B; ε = 1e-3) -> (shares, cash)

Closed-form Cobb-Douglas allocator. Non-preferred assets (γᵢ ≤ 0) get
ε shares; preferred assets get proportional allocation.
"""
function solve_unconstrained_cd_analytical(γ::Vector{Float64}, p::Vector{Float64},
        B::Float64; ε::Float64 = 1e-3)::Tuple{Vector{Float64},Float64}
    K = length(γ)
    preferred = findall(γ .> 0)
    non_preferred = findall(γ .<= 0)
    shares = zeros(K)
    remaining_B = B
    for i in non_preferred
        shares[i] = ε
        remaining_B -= ε * p[i]
    end
    cash = 0.0
    if !isempty(preferred) && remaining_B > 0
        γ_bar = sum(γ[preferred])
        for i in preferred
            shares[i] = (γ[i] / γ_bar) * (remaining_B / p[i])
        end
    else
        cash = remaining_B
    end
    return (shares, cash)
end

"""
    equal_weight_target(K::Int) -> Vector{Float64}

Equal-weight target weights summing to 1.0.
"""
equal_weight_target(K::Int)::Vector{Float64} = fill(1.0 / K, K)

"""
    solve_minvar_buyhold(Σ, bounds) -> Vector{Float64}

Solve min wᵀΣw  s.t.  Σwᵢ = 1, bounds[i,1] ≤ wᵢ ≤ bounds[i,2].
"""
function solve_minvar_buyhold(Σ::Matrix{Float64},
        bounds::Matrix{Float64})::Vector{Float64}
    K = size(Σ, 1)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, w[1:K])
    @constraint(model, [i in 1:K], w[i] >= bounds[i, 1])
    @constraint(model, [i in 1:K], w[i] <= bounds[i, 2])
    @constraint(model, sum(w) == 1.0)
    @objective(model, Min, w' * Σ * w)
    optimize!(model)
    return value.(w)
end
