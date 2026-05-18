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

"""
    run_backtest(strategy, env, cost_model, tax_rates; B₀, rng_seed) -> MyBacktestResult

Walk the hold-out window day by day. `env` is a NamedTuple carrying tickers,
prices (matrix n_days × K), market_prices, volumes, sim_params_init (Dict of
MyEWLSState), σ_m, dates, market_model, c̄.
"""
function run_backtest(strategy::MyAllocationStrategy, env, cost_model::MyCostModel,
        tax_rates::NamedTuple; B₀::Float64 = 100_000.0,
        rng_seed::Int = 42)::MyBacktestResult
    Random.seed!(rng_seed)
    tickers = env.tickers
    K = length(tickers); n_days = size(env.prices, 1)
    Δt = 1.0 / 252.0

    # Initialize state
    state = MyBacktestState()
    state.date_idx = 1
    state.prices = env.prices[1, :]
    state.positions = zeros(Float64, K)
    state.cash = B₀
    state.V_t = B₀
    state.wealth_peak = B₀
    state.sim_state = Dict(t => deepcopy(env.sim_params_init[t]) for t in tickers)
    state.last_decision_t = 0
    state.last_projection = nothing
    state.just_decided = false
    state.next_decision_due = false
    state.trigger_log = MyMPCTrigger[]
    state.trades = NamedTuple[]
    state.ledger = MyTaxLedger()
    state.wealth_after_cost_pretax = zeros(n_days)
    state.wealth_precost_pretax = zeros(n_days)
    state.cash_history = zeros(n_days)
    state.positions_history = zeros(n_days, K)
    gross_cash = B₀

    is_mpc(s::MyAllocationStrategy) = s isa CDWithMPCStrategy || s isa ConstrainedCDWithMPCStrategy

    for t in 1:n_days
        state.date_idx = t
        state.prices = env.prices[t, :]
        state.V_t = sum(state.positions .* state.prices) + state.cash
        gross_V = sum(state.positions .* state.prices) + gross_cash
        state.wealth_peak = max(state.wealth_peak, state.V_t)
        state.just_decided = false

        # Build env_for_step (current γ, Σ from EWLS state)
        αs = [state.sim_state[tk].α for tk in tickers]
        βs = [state.sim_state[tk].β for tk in tickers]
        σ_εs = [state.sim_state[tk].σ_ε for tk in tickers]
        sim_params_now = Dict(tickers[i] => (αs[i], βs[i], σ_εs[i]) for i in 1:K)

        # Rolling market window for gm_t and λ_t
        win = max(1, t - 63)
        mkt_window = env.market_prices[win:t]
        if length(mkt_window) >= 2
            mkt_growth = compute_market_growth(Vector{Float64}(mkt_window))
            gm_t = isempty(mkt_growth) ? 0.0 : mkt_growth[end]
        else
            gm_t = 0.0
        end
        ema_window = Vector{Float64}(mkt_window)
        if length(ema_window) >= 21
            short = compute_ema(ema_window; window = 21)
            long  = compute_ema(ema_window; window = max(21, length(ema_window)))
            λ_t = compute_lambda(short, long)[end]
        else
            λ_t = 0.5
        end
        γ_t = compute_preference_weights(sim_params_now, tickers, gm_t, λ_t)
        ests = MySIMParameterEstimate[]
        for tk in tickers
            e = MySIMParameterEstimate()
            e.ticker = tk; e.α = state.sim_state[tk].α
            e.β = state.sim_state[tk].β; e.σ_ε = state.sim_state[tk].σ_ε; e.r² = 0.5
            push!(ests, e)
        end
        Σ_t = build_sim_covariance(ests, env.σ_m)
        env_step = (γ_t = γ_t, Σ_t = Σ_t, α = αs, β = βs, σ_ε = σ_εs,
                    σ_m = env.σ_m, tickers = tickers,
                    market_model = env.market_model, c̄ = env.c̄)

        if should_decide(strategy, state, t)
            n_target = allocate(strategy, state, t, env_step)
            orders = materialize_orders(tickers, n_target, state.positions,
                state.prices, state.cash, cost_model)
            for o in orders
                idx = findfirst(==(o.ticker), tickers)
                px = state.prices[idx]
                cost = trade_cost(cost_model, o.ticker, o.qty, px)
                state.cash -= cost
                if o.qty > 0
                    state.cash -= o.qty * px
                    gross_cash -= o.qty * px
                    open_lot!(state.ledger, o.ticker, o.qty, px, env.dates[t])
                else
                    sell_qty = -o.qty
                    state.cash += sell_qty * px
                    gross_cash += sell_qty * px
                    close_qty!(state.ledger, o.ticker, sell_qty, px, env.dates[t])
                end
                state.positions[idx] += o.qty
                push!(state.trades, (date = env.dates[t], ticker = o.ticker,
                    qty = o.qty, price = px, cost = cost))
            end
            state.just_decided = true
            state.next_decision_due = false
            if is_mpc(strategy)
                env_proj = merge(env_step, (market_model = env.market_model,))
                state.last_projection = forward_project(state,
                    strategy.spec, env_proj)
                state.last_decision_t = t
            end
        end

        if is_mpc(strategy) && !state.just_decided && state.last_projection !== nothing
            trig = check_trigger(state, strategy.spec)
            push!(state.trigger_log, trig)
            if trig.fired
                state.next_decision_due = true
            end
        end

        # EWLS update every day
        if t >= 2
            g_m_today = log(env.market_prices[t] / env.market_prices[t-1]) / Δt
            for (i, tk) in enumerate(tickers)
                p_prev = env.prices[t-1, i]
                p_now  = env.prices[t, i]
                if p_prev > 0.0 && p_now > 0.0
                    g_i_today = log(p_now / p_prev) / Δt
                    ewls_update!(state.sim_state[tk], g_i_today, g_m_today)
                end
            end
        end

        state.V_t = sum(state.positions .* state.prices) + state.cash
        gross_V = sum(state.positions .* state.prices) + gross_cash
        state.wealth_after_cost_pretax[t] = state.V_t
        state.wealth_precost_pretax[t] = gross_V
        state.cash_history[t] = state.cash
        state.positions_history[t, :] = state.positions
    end

    return build_result(strategy, state, env, tax_rates)
end

function build_result(strategy::MyAllocationStrategy, state::MyBacktestState,
        env, tax_rates::NamedTuple)::MyBacktestResult
    tax_summary = summarize_after_tax(state.ledger, tax_rates)
    wealth_aftertax = copy(state.wealth_after_cost_pretax)
    wealth_aftertax[end] -= tax_summary.total_tax
    return MyBacktestResult(
        strategy_name = string(typeof(strategy).name.name),
        strategy_config = (raw = string(strategy),),
        wealth_after_cost_pretax = state.wealth_after_cost_pretax,
        wealth_after_cost_aftertax = wealth_aftertax,
        wealth_precost_pretax = state.wealth_precost_pretax,
        cash = state.cash_history,
        positions = state.positions_history,
        trades = state.trades,
        trigger_log = state.trigger_log,
        ledger = state.ledger,
        summary = summary_metrics(state, tax_summary))
end

"""
    summary_metrics(state, tax_summary) -> NamedTuple
"""
function summary_metrics(state::MyBacktestState, tax_summary::NamedTuple)::NamedTuple
    W = state.wealth_after_cost_pretax
    n_days = length(W)
    daily_log = [log(W[t+1] / W[t]) for t in 1:n_days-1 if W[t] > 0.0]
    ann_ret = (W[end] / W[1])^(252.0 / n_days) - 1.0
    ann_vol = isempty(daily_log) ? 0.0 : sqrt(252.0) * std(daily_log)
    ann_sharpe = ann_vol > 0.0 ? ann_ret / ann_vol : 0.0
    peak = accumulate(max, W)
    dd = peak .- W
    max_dd = maximum(dd ./ max.(peak, 1e-8))
    turnover_dollar = isempty(state.trades) ? 0.0 : sum(abs(tr.qty * tr.price) for tr in state.trades)
    ann_turnover = turnover_dollar / max(mean(W), 1e-8) * (252.0 / n_days)
    n_trig = count(t.fired for t in state.trigger_log)
    return (
        ann_return = ann_ret,
        ann_volatility = ann_vol,
        ann_sharpe = ann_sharpe,
        max_drawdown = max_dd,
        ann_turnover = ann_turnover,
        lt_share_of_realized = tax_summary.lt_share_of_realized,
        holding_period_median_days = isempty(tax_summary.holding_period_distribution) ?
            0 : Int(round(median(tax_summary.holding_period_distribution))),
        n_mpc_triggers = n_trig)
end
