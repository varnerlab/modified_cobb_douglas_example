# --- estimate_sim and build_sim_covariance vendored from
# eCornell-AI-finance-lectures/code/src/Compute.jl (lines 96-175). ---

"""
    estimate_sim(market_returns, asset_returns, ticker; δ = 0.0) -> MySIMParameterEstimate

Fit α, β, σ_ε via regularized OLS on annualized growth rates.
Inputs and outputs are annualized (1/year units).
"""
function estimate_sim(market_returns::Vector{Float64}, asset_returns::Vector{Float64},
        ticker::String; δ::Float64 = 0.0)::MySIMParameterEstimate
    T = length(market_returns)
    X = hcat(ones(T), market_returns)
    y = asset_returns
    p = 2
    θ̂ = (X' * X + δ * I(p)) \ (X' * y)
    α̂ = θ̂[1]
    β̂ = θ̂[2]
    ŷ = X * θ̂
    residuals = y .- ŷ
    σ_ε = sqrt(dot(residuals, residuals) / (T - p))
    SS_res = dot(residuals, residuals)
    SS_tot = dot(y .- mean(y), y .- mean(y))
    r² = 1.0 - SS_res / SS_tot
    est = MySIMParameterEstimate()
    est.ticker = ticker
    est.α = α̂; est.β = β̂; est.σ_ε = σ_ε; est.r² = r²
    return est
end

"""
    build_sim_covariance(sim_estimates, σ_m) -> Matrix{Float64}

SIM-implied covariance of annualized growth rates:
Σ_ii = β_i² σ_m² + σ_ε_i² ; Σ_ij = β_i β_j σ_m².
"""
function build_sim_covariance(sim_estimates::Vector{MySIMParameterEstimate},
        σ_m::Float64)::Matrix{Float64}
    N = length(sim_estimates)
    Σ = zeros(N, N)
    σ_m² = σ_m^2
    for i ∈ 1:N
        βᵢ = sim_estimates[i].β
        σ_εᵢ = sim_estimates[i].σ_ε
        for j ∈ 1:N
            βⱼ = sim_estimates[j].β
            Σ[i, j] = (i == j) ? βᵢ^2 * σ_m² + σ_εᵢ^2 : βᵢ * βⱼ * σ_m²
        end
    end
    return Σ
end
