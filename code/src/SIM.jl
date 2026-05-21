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
    build_sim_covariance(sim_estimates, σ_m; Δt = 1/252) -> Matrix{Float64}

SIM-implied covariance of TRUE annualized growth rates:
Σ_ii = (β_i² σ_m² + σ_ε_i²) · Δt ; Σ_ij = β_i · β_j · σ_m² · Δt.

**Units convention** (the load-bearing detail). `σ_m` and `σ_ε` here are the
SDs of `compute_market_growth(prices; Δt)` output — i.e., per-day log-returns
annualized by `/Δt`. Their variance is therefore `(1/Δt)² × var(per_step) =
(1/Δt) × var_annualized`, so they exceed the *true* annualized SD by √(1/Δt)
(= √252 at daily Δt). The `· Δt` factor inside this function converts the
inflated variance back to true annualized variance, which is what every
downstream consumer assumes:

  - `forward_project_closed_form` computes σ² = dot(w, Σ * w) × τ × Δt; with
    Σ in true annualized variance this gives the variance of V_τ over τ
    trading days (Δt = 1/252 years per step), matching the MC arm.
  - σ_max in the constrained-CD allocator is documented as an annualized
    portfolio vol cap; with Σ in true annualized variance, √(w'Σw) ≤ σ_max
    correctly bounds the annualized SD.

Pre-2026-05-21 the function omitted the `Δt` factor, so Σ was inflated by
1/Δt = 252; that made σ_max bind at √252× tighter than its nominal value
(0.76% true annualized vs. the 12% documented) and made the closed-form
arm of `forward_project` overstate horizon σ by √252×. See
`lambda_swap_note.md` for the full audit.
"""
function build_sim_covariance(sim_estimates::Vector{MySIMParameterEstimate},
        σ_m::Float64; Δt::Float64 = 1.0/252.0)::Matrix{Float64}
    N = length(sim_estimates)
    Σ = zeros(N, N)
    σ_m² = σ_m^2 * Δt
    for i ∈ 1:N
        βᵢ = sim_estimates[i].β
        σ_εᵢ = sim_estimates[i].σ_ε
        σ_εᵢ² = σ_εᵢ^2 * Δt
        for j ∈ 1:N
            βⱼ = sim_estimates[j].β
            Σ[i, j] = (i == j) ? βᵢ^2 * σ_m² + σ_εᵢ² : βᵢ * βⱼ * σ_m²
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
    compute_lambda(short_ema, long_ema; G = 1.0) -> Vector{Float64}

Signed regime-lens λ from an EMA crossover sentiment signal:

    λ_t = -G · (short_ema_t / long_ema_t - 1)

with gain G > 0. Sign convention: λ_t > 0 is bearish (short below long; risk-averse),
λ_t < 0 is bullish (short above long; take more risk), λ_t ≈ 0 is neutral. G is a
sensitivity hyperparameter that must be calibrated to the operational regime — see
lambda_swap_note.md at the repo root for the rationale behind the signed form.

The returned vector is a time series of λ values aligned with the input EMA series
(one λ per timestep); callers index it at the decision time of interest.
"""
function compute_lambda(short_ema::Vector{Float64}, long_ema::Vector{Float64};
        G::Float64 = 1.0)::Vector{Float64}
    @assert length(short_ema) == length(long_ema)
    return -G .* (short_ema ./ long_ema .- 1.0)
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
        # 1e-8 floor guards division stability when abs(βᵢ)^lambda → 0
        # (small βᵢ with lambda > 0). For signed lambda < 0 the exponent flips and
        # abs(βᵢ)^lambda blows up instead — that drives α/RF → 0 harmlessly.
        RF = max(abs(βᵢ)^lambda, 1e-8)
        g_hat = αᵢ / RF + (abs(βᵢ)^(1.0 - lambda)) * gm_t
        γ[i] = tanh(g_hat)
    end
    return γ
end

# --- EWLS (vendored from Compute.jl lines 3108-3177). ---

"""
    ewls_init(α₀, β₀, σ_ε₀; half_life = 63.0, prior_weight = 63.0) -> MyEWLSState

Initialize EWLS state from a prior (α₀, β₀, σ_ε₀). Decay η = 2^(-1/half_life);
prior_weight seeds the sufficient statistics so EWLS recovers (α₀, β₀) exactly
on day 0.
"""
function ewls_init(α₀::Float64, β₀::Float64, σ_ε₀::Float64;
        half_life::Float64 = 63.0, prior_weight::Float64 = 63.0)::MyEWLSState
    η = 2.0^(-1.0 / half_life)
    s = MyEWLSState()
    s.Sw   = prior_weight
    s.Swx  = 0.0
    s.Swy  = prior_weight * α₀
    s.Swxx = prior_weight * 1.0
    s.Swxy = prior_weight * β₀
    s.Swyy = prior_weight * (α₀^2 + β₀^2 + σ_ε₀^2)
    s.η = η
    s.α = α₀; s.β = β₀; s.σ_ε = σ_ε₀
    return s
end

"""
    ewls_update!(state, g_i, g_m) -> (α, β, σ_ε)

Decay running sums by η, add new (g_i, g_m) observation with unit weight,
recompute (α, β, σ_ε). Returns the updated estimates.
"""
function ewls_update!(state::MyEWLSState, g_i::Float64,
        g_m::Float64)::Tuple{Float64,Float64,Float64}
    η = state.η
    state.Sw   = η * state.Sw   + 1.0
    state.Swx  = η * state.Swx  + g_m
    state.Swy  = η * state.Swy  + g_i
    state.Swxx = η * state.Swxx + g_m * g_m
    state.Swxy = η * state.Swxy + g_i * g_m
    state.Swyy = η * state.Swyy + g_i * g_i
    denom = state.Sw * state.Swxx - state.Swx^2
    if abs(denom) > 1e-12
        state.β = (state.Sw * state.Swxy - state.Swx * state.Swy) / denom
        state.α = (state.Swy - state.β * state.Swx) / state.Sw
        mse = (state.Swyy
               - 2.0 * state.β * state.Swxy
               - 2.0 * state.α * state.Swy
               + state.β^2 * state.Swxx
               + 2.0 * state.α * state.β * state.Swx
               + state.α^2 * state.Sw) / state.Sw
        state.σ_ε = sqrt(max(mse, 0.0))
    end
    return (state.α, state.β, state.σ_ε)
end
