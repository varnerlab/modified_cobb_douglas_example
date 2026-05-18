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
    duals::NamedTuple
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

abstract type MyAllocationStrategy end

struct EqualWeightStrategy <: MyAllocationStrategy end
struct MinVarBuyHoldStrategy <: MyAllocationStrategy end
struct UnconstrainedCDStrategy <: MyAllocationStrategy end

Base.@kwdef struct CostAwareMVStrategy <: MyAllocationStrategy
    κ::Float64
    c::Float64
end

Base.@kwdef struct CDWithMPCStrategy <: MyAllocationStrategy
    spec::MyMPCSpec
end

Base.@kwdef struct ConstrainedCDWithMPCStrategy <: MyAllocationStrategy
    spec::MyMPCSpec
    σ_max::Float64
    K_turnover::Float64
    w_max::Float64
end

# --- Bandit ---

Base.@kwdef struct MyBanditConfig
    K_basket::Int = 22
    iters_per_arm::Int = 50
    iters_max::Int = 5000
    iters_min::Int = 500
    ε_floor::Float64 = 0.05
    forward_horizon::Int = 21
    seed::Int = 2026
end

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
