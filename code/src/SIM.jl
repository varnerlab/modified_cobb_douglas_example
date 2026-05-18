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

# --- compute_market_growth, compute_ema, compute_lambda vendored
# from Compute.jl lines 360-460. ---

"""
    compute_market_growth(prices; Δt = 1/252) -> Vector{Float64}

Annualized log-returns of a price series.
"""
function compute_market_growth(prices::Vector{Float64};
        Δt::Float64 = 1.0 / 252.0)::Vector{Float64}
    T = length(prices)
    g = zeros(T - 1)
    for t ∈ 1:T-1
        g[t] = log(prices[t+1] / prices[t]) / Δt
    end
    return g
end

"""
    compute_ema(prices; window = 21) -> Vector{Float64}

Exponential moving average with span = window.
"""
function compute_ema(prices::Vector{Float64}; window::Int = 21)::Vector{Float64}
    α = 2.0 / (window + 1.0)
    ema = similar(prices)
    ema[1] = prices[1]
    for t ∈ 2:length(prices)
        ema[t] = α * prices[t] + (1.0 - α) * ema[t-1]
    end
    return ema
end

"""
    compute_lambda(short_ema, long_ema; θ = 0.5) -> Vector{Float64}

Regime-lens λ from EMA crossover. λ_t = 1 / (1 + exp(-(short-long)/θ)) (sigmoid).
"""
function compute_lambda(short_ema::Vector{Float64}, long_ema::Vector{Float64};
        θ::Float64 = 0.5)::Vector{Float64}
    @assert length(short_ema) == length(long_ema)
    diff = (short_ema .- long_ema) ./ θ
    return 1.0 ./ (1.0 .+ exp.(-diff))
end

"""
    compute_preference_weights(sim_parameters, tickers, gm_t, lambda) -> Vector{Float64}

No-news variant of the lectures function (spec §3.3).
γ_i = tanh(α_i/|β|^λ + |β|^(1-λ) · gm_t)
"""
function compute_preference_weights(
        sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}},
        tickers::Vector{String}, gm_t::Float64, lambda::Float64)::Vector{Float64}
    K = length(tickers)
    γ = zeros(K)
    for i in 1:K
        (αᵢ, βᵢ, _) = sim_parameters[tickers[i]]
        RF = max(abs(βᵢ)^lambda, 1e-8)
        g_hat = αᵢ / RF + (abs(βᵢ)^(1.0 - lambda)) * gm_t
        γ[i] = tanh(g_hat)
    end
    return γ
end
