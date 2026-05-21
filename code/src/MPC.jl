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
    # σ_m and σ_ε come from EWLS / JumpHMM fits on compute_market_growth output,
    # which scales daily log-returns by 1/Δt. Their variance is therefore 1/Δt ×
    # true annualized variance; the · Δt factor converts back so Σ is in true
    # annualized variance — matching the units the closed-form formulas
    # (μ_log = μ_per_step·τ·Δt, σ² = σ²_per_step·τ·Δt) assume. See
    # `build_sim_covariance` in SIM.jl for the full rationale.
    Σ = zeros(K, K)
    σ_m² = σ_m^2 * Δt
    for i in 1:K, j in 1:K
        Σ[i, j] = (i == j) ? β[i]^2 * σ_m² + σ_ε[i]^2 * Δt :
                             β[i] * β[j] * σ_m²
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
    forward_project(state, spec, env; rng = Random.default_rng()) -> MyMPCProjection

JumpHMM-SIM hybrid forward projection (spec §5.1) plus closed-form arms.

`env` is a NamedTuple carrying `market_model::JumpHiddenMarkovModel`,
plus per-ticker `α`, `β`, `σ_ε` vectors (in the same order as `state.positions`),
plus `σ_m::Float64`, plus `tickers::Vector{String}`. The `Δt` is fixed at 1/252.

`rng` is threaded through the per-asset idiosyncratic noise draws so that
parallel callers (e.g. `compare_strategies(...; parallel = true)`) do not
contend on the global RNG. `hmm_simulate` still uses Julia's default RNG.
"""
function forward_project(state, spec::MyMPCSpec, env;
        rng::AbstractRNG = Random.default_rng())::MyMPCProjection
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
                g_i = α[i] + β[i] * G_market[τ] + σ_ε[i] * randn(rng)
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

Four conditions (any one fires):
  1. `state.last_alloc_was_cash` AND τ ≥ spec.cash_revisit_interval
     (defensive cash regime — re-evaluate after the configured interval)
  2. drawdown > spec.D_max (circuit breaker)
  3. τ >= spec.T (horizon refresh)
  4. V_t outside band [μ_τ ± z·σ_τ]

When `cash_revisit_interval` is left at its default (= T = 21), the cash
fast-path races horizon-elapsed and is observed as `:cash_revisit` in the
trigger log instead of `:horizon_elapsed`; the cadence is the same. Setting
`cash_revisit_interval` shorter than T lets the strategy re-evaluate the
defensive ε-pin regime on a faster cadence than the held-position horizon.
A smoke test on the canonical seed shows that interval = 1 (daily revisit)
hurts Sharpe and worsens drawdown by re-entering deteriorating regimes; the
parameter is a knob, not a free improvement.
"""
function check_trigger(state, spec::MyMPCSpec)::MyMPCTrigger
    proj = state.last_projection
    t_global = state.date_idx
    τ = state.date_idx - state.last_decision_t
    if proj === nothing || τ <= 0
        return MyMPCTrigger(fired = false, reason = :in_spec,
                            τ = max(τ, 0), t_global = t_global)
    end
    # Cash-revisit fast-path: when the prior fire returned the ε-pin defensive
    # regime, re-evaluate after `cash_revisit_interval` trading days rather
    # than waiting for the full T-day horizon. Default interval = T preserves
    # the original held-position cadence (fires race horizon-elapsed and win).
    if state.last_alloc_was_cash && τ >= spec.cash_revisit_interval
        return MyMPCTrigger(fired = true, reason = :cash_revisit,
                            τ = τ, t_global = t_global)
    end
    # Drawdown — circuit breaker
    if state.wealth_peak > 0.0
        dd = (state.wealth_peak - state.V_t) / state.wealth_peak
        if dd > spec.D_max
            return MyMPCTrigger(fired = true, reason = :drawdown,
                                τ = τ, t_global = t_global)
        end
    end
    if τ >= spec.T
        return MyMPCTrigger(fired = true, reason = :horizon_elapsed,
                            τ = τ, t_global = t_global)
    end
    τ_clamped = min(τ, length(proj.μ))
    μτ = proj.μ[τ_clamped]; στ = proj.σ[τ_clamped]
    if state.V_t < μτ - spec.z * στ || state.V_t > μτ + spec.z * στ
        return MyMPCTrigger(fired = true, reason = :band_exit,
                            τ = τ, t_global = t_global)
    end
    return MyMPCTrigger(fired = false, reason = :in_spec,
                        τ = τ, t_global = t_global)
end
