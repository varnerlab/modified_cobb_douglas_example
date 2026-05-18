# MPC module — forward projection (JumpHMM-SIM hybrid MC + closed-form lognormal arm).

"""
    forward_project_closed_form(α, β, σ_m, σ_ε, w, V₀, T, Δt) -> (μ, σ)

Closed-form lognormal forward projection (spec §5.2). Returns vectors
of mean and std of V_τ for τ = 1..T under constant-weight GBM.
"""
function forward_project_closed_form(α::Vector{Float64}, β::Vector{Float64},
        σ_m::Float64, σ_ε::Vector{Float64}, w::Vector{Float64},
        V₀::Float64, T::Int, Δt::Float64)::Tuple{Vector{Float64},Vector{Float64}}
    K = length(α)
    Σ = zeros(K, K)
    σ_m² = σ_m^2
    for i in 1:K, j in 1:K
        Σ[i, j] = (i == j) ? β[i]^2 * σ_m² + σ_ε[i]^2 : β[i] * β[j] * σ_m²
    end
    μ_log_per_step = dot(w, α) - 0.5 * dot(w, Σ * w)   # Itô-corrected
    σ²_per_step    = dot(w, Σ * w)
    μ = zeros(T); σ = zeros(T)
    for τ in 1:T
        μ_log = μ_log_per_step * τ * Δt
        σ²    = σ²_per_step    * τ * Δt
        # Lognormal mean and std of V_τ:
        μ[τ] = V₀ * exp(μ_log + 0.5 * σ²)
        σ[τ] = μ[τ] * sqrt(max(exp(σ²) - 1.0, 0.0))
    end
    return (μ, σ)
end

"""
    forward_project(state, spec, env) -> MyMPCProjection

JumpHMM-SIM hybrid forward projection (spec §5.1) plus closed-form arms.

`env` is a NamedTuple carrying `market_model::JumpHiddenMarkovModel`,
plus per-ticker `α`, `β`, `σ_ε` vectors (in the same order as `state.positions`),
plus `σ_m::Float64`, plus `tickers::Vector{String}`. The `Δt` is fixed at 1/252.
"""
function forward_project(state, spec::MyMPCSpec, env)::MyMPCProjection
    Δt = 1.0 / 252.0
    T = spec.T; N = spec.N
    n = state.positions
    prices0 = state.prices
    V₀ = sum(n .* prices0) + state.cash
    K = length(n)
    α = env.α; β = env.β; σ_ε = env.σ_ε; σ_m = env.σ_m

    # MC paths
    sim_result = hmm_simulate(env.market_model, T; n_paths = N)
    paths = zeros(N, T)
    for j in 1:N
        G_full = Float64.(sim_result.paths[j].observations)
        G_market = length(G_full) > T ? G_full[2:T+1] : G_full[1:T]
        if length(G_market) < T
            G_market = vcat(G_market, zeros(T - length(G_market)))
        end
        # propagate prices for each asset along this market path
        P = copy(prices0)
        for τ in 1:T
            for i in 1:K
                g_i = α[i] + β[i] * G_market[τ] + σ_ε[i] * randn()
                P[i] = P[i] * exp(g_i * Δt)
            end
            paths[j, τ] = sum(n .* P) + state.cash
        end
    end
    μ_arr = vec(mean(paths; dims = 1))
    σ_arr = vec(std(paths;  dims = 1))

    # Closed-form arm — uses current weights at decision time
    w_now = (n .* prices0) ./ V₀
    cf_μ, cf_σ = forward_project_closed_form(α, β, σ_m, σ_ε, w_now, V₀, T, Δt)

    # Divergence flag
    div_warn = any(τ -> begin
        rel = abs(σ_arr[τ] - cf_σ[τ]) / max(cf_σ[τ], 1e-8)
        rel > 0.25
    end, 1:T)

    return MyMPCProjection(
        μ = μ_arr, σ = σ_arr,
        V₀ = V₀, paths = paths,
        decision_date_idx = state.date_idx,
        closed_form_μ = cf_μ,
        closed_form_σ = cf_σ,
        divergence_warning = div_warn)
end

"""
    check_trigger(state, spec::MyMPCSpec) -> MyMPCTrigger

Three conditions (any one fires):
  1. V_t outside band [μ_τ ± z·σ_τ]
  2. τ >= spec.T (horizon refresh)
  3. drawdown > spec.D_max (circuit breaker)
"""
function check_trigger(state, spec::MyMPCSpec)::MyMPCTrigger
    proj = state.last_projection
    τ = state.date_idx - state.last_decision_t
    if proj === nothing || τ <= 0
        return MyMPCTrigger(fired = false, reason = :in_spec, τ = max(τ, 0))
    end
    # Drawdown first — circuit breaker
    if state.wealth_peak > 0.0
        dd = (state.wealth_peak - state.V_t) / state.wealth_peak
        if dd > spec.D_max
            return MyMPCTrigger(fired = true, reason = :drawdown, τ = τ)
        end
    end
    if τ >= spec.T
        return MyMPCTrigger(fired = true, reason = :horizon_elapsed, τ = τ)
    end
    τ_clamped = min(τ, length(proj.μ))
    μτ = proj.μ[τ_clamped]; στ = proj.σ[τ_clamped]
    if state.V_t < μτ - spec.z * στ || state.V_t > μτ + spec.z * στ
        return MyMPCTrigger(fired = true, reason = :band_exit, τ = τ)
    end
    return MyMPCTrigger(fired = false, reason = :in_spec, τ = τ)
end
