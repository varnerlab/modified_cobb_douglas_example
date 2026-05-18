# Types module — struct definitions for the constrained Cobb-Douglas + MPC system.

# --- SIM parameter estimates ---

"""
    MySIMParameterEstimate

Single-index-model parameters for one asset, on annualized growth rates.
"""
mutable struct MySIMParameterEstimate
    ticker::String
    α::Float64
    β::Float64
    σ_ε::Float64
    r²::Float64
    MySIMParameterEstimate() = new()
end

"""
    MyEWLSState

Exponentially-weighted least-squares state for online SIM updating.
Sufficient statistics `Sw…Swyy` decay by `η` per step; (`α`, `β`, `σ_ε`)
are the running point estimates.
"""
mutable struct MyEWLSState
    Sw::Float64; Swx::Float64; Swy::Float64
    Swxx::Float64; Swxy::Float64; Swyy::Float64
    η::Float64
    α::Float64; β::Float64; σ_ε::Float64
    MyEWLSState() = new()
end

# --- Allocator I/O ---

"""
    MyConstrainedCDProblem

Inputs to the constrained Cobb-Douglas solver. See spec §4.
"""
Base.@kwdef struct MyConstrainedCDProblem
    γ::Vector{Float64}
    p::Vector{Float64}
    B::Float64
    Σ::Matrix{Float64}
    σ_max::Float64
    K_turnover::Float64
    w_max::Float64
    n_prev::Vector{Float64}
    c̄::Float64
end

"""
    MyConstrainedCDResult

Output of the constrained Cobb-Douglas solver.
"""
Base.@kwdef struct MyConstrainedCDResult
    n::Vector{Float64}
    w::Vector{Float64}
    unallocated_budget::Float64
    status::Symbol
    objective::Float64
end

# --- MPC layer I/O ---

"""
    MyMPCSpec

MPC trigger configuration. `z` = band z-score, `T` = horizon in trading days,
`N` = number of MC paths, `D_max` = circuit-breaker drawdown.
"""
Base.@kwdef struct MyMPCSpec
    z::Float64 = 1.96
    T::Int = 21
    N::Int = 1000
    D_max::Float64 = 0.08
end

"""
    MyMPCProjection

Forward-projection output: mean / std bands + paths + closed-form
validation arms.
"""
Base.@kwdef struct MyMPCProjection
    μ::Vector{Float64}
    σ::Vector{Float64}
    V₀::Float64
    paths::Matrix{Float64}             # N paths × T steps
    decision_date_idx::Int
    closed_form_μ::Vector{Float64}
    closed_form_σ::Vector{Float64}
    divergence_warning::Bool
end

"""
    MyMPCTrigger

Outcome of a single `check_trigger` call.
"""
Base.@kwdef struct MyMPCTrigger
    fired::Bool
    reason::Symbol   # :band_exit, :horizon_elapsed, :drawdown, :in_spec
    τ::Int
end

# --- Cost model ---

"""
    MyCostModel

Commission / half-spread / slippage parameters plus per-ticker ADV.
"""
Base.@kwdef struct MyCostModel
    commission_per_trade::Float64 = 0.0
    half_spread_bps::Float64 = 5.0
    slippage_κ::Float64 = 0.001
    adv::Dict{String,Float64}
end

# --- Tax ---

"""
    MyTaxLot

A single open tax lot (FIFO queue element).
"""
Base.@kwdef mutable struct MyTaxLot
    ticker::String
    open_date::Date
    open_price::Float64
    qty::Int
end

"""
    MyTaxLedger

Lot-by-lot FIFO ledger. `lots[ticker]` is the FIFO queue (front = oldest);
`closed_lots` accumulates realizations for diagnostics.
"""
mutable struct MyTaxLedger
    lots::Dict{String,Vector{MyTaxLot}}
    closed_lots::Vector{NamedTuple}
    realized_st_pnl::Float64
    realized_lt_pnl::Float64
    MyTaxLedger() = new(Dict{String,Vector{MyTaxLot}}(), NamedTuple[], 0.0, 0.0)
end

# --- Strategy abstraction (spec §2.1) ---

"""
    MyAllocationStrategy

Abstract supertype for all allocation strategies in the backtest harness.
Subtypes `S <: MyAllocationStrategy` must implement two dispatch methods:
`allocate(::S, state, t, env)` returning target positions/weights, and
`should_decide(::S, state, t)` returning a `Bool` that gates when the
allocator runs on day `t`.
"""
abstract type MyAllocationStrategy end

"""
    EqualWeightStrategy

Strategy 1 (spec §6.2): passive equal-weight buy-and-hold on the assembled
basket. Allocates once at `t = 1` with `wᵢ = 1/N` and holds for the entire
horizon (no rebalance, no MPC).
"""
struct EqualWeightStrategy <: MyAllocationStrategy end

"""
    MinVarBuyHoldStrategy

Strategy 2 (spec §6.2): minimum-variance portfolio computed at `t = 1` from
the SIM-implied covariance Σ, then held without rebalance. Long-only,
fully-invested baseline that isolates the value of variance-minimization
without dynamics.
"""
struct MinVarBuyHoldStrategy <: MyAllocationStrategy end

"""
    UnconstrainedCDStrategy

Strategy 3 (spec §6.2): analytical Cobb-Douglas allocator (closed-form
`wᵢ ∝ γᵢ`) rebalanced daily. Replicates the live engine's allocator at a
daily cadence and serves as the unconstrained, no-trigger benchmark for the
constrained / MPC-gated variants.
"""
struct UnconstrainedCDStrategy <: MyAllocationStrategy end

"""
    CostAwareMVStrategy

Strategy 4 (spec §6.2): mean-variance allocator with a γ-tilt in the
expected-return term and an l1 turnover penalty in the objective. Field
`κ` is the risk-aversion coefficient on the quadratic variance term;
`c` is the per-unit l1 turnover cost.
"""
Base.@kwdef struct CostAwareMVStrategy <: MyAllocationStrategy
    κ::Float64
    c::Float64
end

"""
    CDWithMPCStrategy

Strategy 5 (spec §6.2): unconstrained Cobb-Douglas allocator gated by the
MPC trigger. Re-allocates only when the MPC band fires (band exit, horizon
elapsed, or drawdown circuit-breaker); otherwise holds. Field `spec` is
the [`MyMPCSpec`](@ref) trigger configuration.
"""
Base.@kwdef struct CDWithMPCStrategy <: MyAllocationStrategy
    spec::MyMPCSpec
end

"""
    ConstrainedCDWithMPCStrategy

Strategy 6 (spec §6.2): the new design under evaluation. Cobb-Douglas
allocator subject to a covariance cap (`σ_max`), l1 turnover budget
(`K_turnover`), and per-name concentration cap (`w_max`), gated by the
MPC trigger (`spec`). Combines the variance/turnover/concentration
constraints of Strategy 4 with the event-driven cadence of Strategy 5.
"""
Base.@kwdef struct ConstrainedCDWithMPCStrategy <: MyAllocationStrategy
    spec::MyMPCSpec
    σ_max::Float64
    K_turnover::Float64
    w_max::Float64
end

# --- Bandit ---

"""
    MyBanditConfig

Configuration for the per-sector ε-greedy bandit that assembles the trading
basket. Controls basket size (`K_basket`), per-arm and total iteration
caps (`iters_per_arm`, `iters_max`, `iters_min`), exploration floor
(`ε_floor`), the forward evaluation window (`forward_horizon`), and the
RNG `seed` for reproducibility.
"""
Base.@kwdef struct MyBanditConfig
    K_basket::Int = 22
    iters_per_arm::Int = 50
    iters_max::Int = 5000
    iters_min::Int = 500
    ε_floor::Float64 = 0.05
    forward_horizon::Int = 21
    seed::Int = 2026
end

"""
    MyBanditResult

Output of one bandit training run: per-sector ticker `quotas`, the winning
arm indices and mean rewards (`sector_best_arms`, `sector_best_means`),
the full reward history per sector (`sector_reward_history`), the
assembled basket as both tickers and integer indices
(`full_basket_tickers`, `full_basket_indices`), the `seed` used, and a
`holdout_metrics` NamedTuple of out-of-sample diagnostics.
"""
Base.@kwdef struct MyBanditResult
    quotas::Dict{String,Int}
    sector_best_arms::Dict{String,Vector{Int}}
    sector_best_means::Dict{String,Float64}
    sector_reward_history::Dict{String,Vector{Float64}}
    full_basket_tickers::Vector{String}
    full_basket_indices::Vector{Int}
    seed::Int
    holdout_metrics::NamedTuple
end

# --- Backtest harness ---

"""
    MyBacktestState

Mutable per-day state threaded through the harness loop. Holds the current
prices/positions/cash and portfolio value, the wealth peak (for drawdown
checks), per-ticker EWLS SIM state, the MPC decision bookkeeping
(`last_decision_t`, `last_projection`, `just_decided`, `next_decision_due`),
and the running history vectors for trigger log, trades, tax ledger, and
wealth/cash/positions series.
"""
mutable struct MyBacktestState
    date_idx::Int
    prices::Vector{Float64}
    positions::Vector{Float64}
    cash::Float64
    V_t::Float64
    wealth_peak::Float64
    sim_state::Dict{String,MyEWLSState}
    last_decision_t::Int
    last_projection::Union{Nothing,MyMPCProjection}
    just_decided::Bool
    next_decision_due::Bool
    trigger_log::Vector{MyMPCTrigger}
    trades::Vector{NamedTuple}
    ledger::MyTaxLedger
    wealth_after_cost_pretax::Vector{Float64}
    wealth_precost_pretax::Vector{Float64}
    cash_history::Vector{Float64}
    positions_history::Matrix{Float64}
    MyBacktestState() = new()
end

"""
    MyBacktestResult

Frozen per-strategy output of a backtest: identifying `strategy_name` and
`strategy_config`, three wealth series
(`wealth_after_cost_pretax`, `wealth_after_cost_aftertax`,
`wealth_precost_pretax`) for cost/tax attribution, the `cash` series and
`positions` matrix over time, the list of executed `trades`, the
`trigger_log` of MPC decisions, the closed/open tax `ledger`, and a
`summary` NamedTuple of headline metrics.
"""
Base.@kwdef struct MyBacktestResult
    strategy_name::String
    strategy_config::NamedTuple
    wealth_after_cost_pretax::Vector{Float64}
    wealth_after_cost_aftertax::Vector{Float64}
    wealth_precost_pretax::Vector{Float64}
    cash::Vector{Float64}
    positions::Matrix{Float64}
    trades::Vector{NamedTuple}
    trigger_log::Vector{MyMPCTrigger}
    ledger::MyTaxLedger
    summary::NamedTuple
end
