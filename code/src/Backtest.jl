"""
    materialize_orders(tickers, n_target, n_current, prices, cash_available,
                       cost_model; min_dollar = 100.0) -> Vector{NamedTuple}

Round target shares to integers and emit only orders whose absolute dollar
value exceeds `min_dollar`. The min-order threshold defuses the live engine's
γ-jitter problem (spec §1.3) by suppressing sub-threshold churn.

Returns a vector of (ticker::String, qty::Int) NamedTuples (qty > 0 = buy,
qty < 0 = sell). Cost handling happens at execution time, not here.
"""
function materialize_orders(tickers::Vector{String}, n_target::Vector{Float64},
        n_current::Vector{Float64}, prices::Vector{Float64},
        cash_available::Float64, cost_model::MyCostModel;
        min_dollar::Float64 = 100.0)::Vector{NamedTuple}
    orders = NamedTuple[]
    K = length(tickers)
    for i in 1:K
        q = Int(round(n_target[i] - n_current[i]))
        if q == 0
            continue
        end
        notional = abs(q) * prices[i]
        if notional < min_dollar
            continue
        end
        push!(orders, (ticker = tickers[i], qty = q))
    end
    return orders
end

# --- Strategy dispatch ---

should_decide(::EqualWeightStrategy, state::MyBacktestState, t::Int)::Bool = (t == 1)
should_decide(::MinVarBuyHoldStrategy, state::MyBacktestState, t::Int)::Bool = (t == 1)
should_decide(::UnconstrainedCDStrategy, state::MyBacktestState, t::Int)::Bool = true
should_decide(::CostAwareMVStrategy, state::MyBacktestState, t::Int)::Bool = true
should_decide(::CDWithMPCStrategy, state::MyBacktestState, t::Int)::Bool =
    (t == 1) || state.next_decision_due
should_decide(::ConstrainedCDWithMPCStrategy, state::MyBacktestState, t::Int)::Bool =
    (t == 1) || state.next_decision_due

"""
    allocate(strategy, state, t, env) -> Vector{Float64}

Return target share counts (length K). `env` carries γ_t, Σ_t, prices, B, n_prev
information the strategy needs.
"""
function allocate(::EqualWeightStrategy, state::MyBacktestState, t::Int, env)::Vector{Float64}
    K = length(state.prices)
    w = equal_weight_target(K)
    B = state.V_t
    return (w .* B) ./ state.prices
end

function allocate(::MinVarBuyHoldStrategy, state::MyBacktestState, t::Int, env)::Vector{Float64}
    K = length(state.prices)
    bounds = [zeros(K) ones(K)]
    w = solve_minvar_buyhold(env.Σ_t, bounds)
    B = state.V_t
    return (w .* B) ./ state.prices
end

function allocate(::UnconstrainedCDStrategy, state::MyBacktestState, t::Int, env)::Vector{Float64}
    B = state.V_t
    n, _ = solve_unconstrained_cd_analytical(env.γ_t, state.prices, B)
    return n
end

function allocate(s::CostAwareMVStrategy, state::MyBacktestState, t::Int, env)::Vector{Float64}
    B = state.V_t
    w_prev = (state.positions .* state.prices) ./ max(B, 1e-8)
    w = solve_cost_aware_mv(env.γ_t, env.Σ_t, w_prev; κ = s.κ, c = s.c)
    return (w .* B) ./ state.prices
end

function allocate(::CDWithMPCStrategy, state::MyBacktestState, t::Int, env)::Vector{Float64}
    B = state.V_t
    n, _ = solve_unconstrained_cd_analytical(env.γ_t, state.prices, B)
    return n
end

function allocate(s::ConstrainedCDWithMPCStrategy, state::MyBacktestState, t::Int, env)::Vector{Float64}
    B = state.V_t
    problem = MyConstrainedCDProblem(
        γ = env.γ_t, p = state.prices, B = B, Σ = env.Σ_t,
        σ_max = s.σ_max, K_turnover = s.K_turnover, w_max = s.w_max,
        n_prev = state.positions, c̄ = env.c̄)
    res = solve_constrained_cd(problem)
    if res.status == :no_preferred || res.status == :solver_failed
        return copy(state.positions)
    end
    return res.n
end
