# Constrained Cobb-Douglas + MPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained Julia package + 5-script pipeline + Jupyter notebook + Documenter.jl-published docs that implement the constrained Cobb-Douglas + MPC strategy from `constrained_cobb_douglas.md`, with a 6-strategy bake-off on the 2025-2026 S&P 500 hold-out.

**Architecture:** Modular Julia package (`ConstrainedCobbDouglas`) with one module per concern (Types, SIM, Allocator, MPC, Costs, Tax, Bandit, Backtest, Files). Strategy dispatch via abstract type. Scripts produce JLD2 artifacts; notebook reads and renders. Docs auto-deploy to GitHub Pages.

**Tech Stack:** Julia 1.10+, JuMP, Clarabel (SCS fallback), JumpHMM, JLD2, CSV, DataFrames, Documenter.jl, IJulia (notebook), Test.jl.

**Spec:** `docs/superpowers/specs/2026-05-17-constrained-cobb-douglas-design.md`

---

## Phase 0 — Package bootstrap

### Task 1: Initialize the Julia package skeleton

**Files:**
- Create: `code/Project.toml`
- Create: `code/src/ConstrainedCobbDouglas.jl`
- Create: `code/test/runtests.jl`
- Create: `.gitignore`

- [ ] **Step 1: Create `code/Project.toml`**

```toml
name = "ConstrainedCobbDouglas"
uuid = "b2c3d4e5-f6a7-8901-bcde-f12345678902"
version = "0.1.0"

[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Clarabel = "61c947e1-3e6d-4ee4-985a-eec8c727bd6e"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
JumpHMM = "6ee30eab-673b-484c-8b30-2cdb31ee1eb8"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SCS = "c946c3f1-0d1f-5ce8-9dea-7daa1f7e2d13"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[sources]
JumpHMM = {rev = "main", url = "https://github.com/varnerlab/JumpHMM.jl.git"}

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```

- [ ] **Step 2: Create umbrella module `code/src/ConstrainedCobbDouglas.jl`**

```julia
module ConstrainedCobbDouglas

using Clarabel
using CSV
using DataFrames
using Dates
using Distributions
using FileIO
using JLD2
using JuMP
using JumpHMM
using LinearAlgebra
using Random
using SCS
using Statistics
using StatsBase

const _PATH_TO_SRC = dirname(@__FILE__)
const _PATH_TO_DATA = joinpath(_PATH_TO_SRC, "data")

const hmm_simulate = JumpHMM.simulate

# Module includes (populated in later tasks)
include("Types.jl")
# include("SIM.jl")
# include("Allocator.jl")
# include("MPC.jl")
# include("Costs.jl")
# include("Tax.jl")
# include("Bandit.jl")
# include("Backtest.jl")
# include("Files.jl")

end # module
```

- [ ] **Step 3: Create empty placeholder `code/src/Types.jl`**

```julia
# Types module — struct definitions for the constrained Cobb-Douglas + MPC system.
# Populated in Task 4.
```

- [ ] **Step 4: Create `code/test/runtests.jl`**

```julia
using Test
using ConstrainedCobbDouglas

@testset "ConstrainedCobbDouglas.jl" begin
    # Test files will be included in later tasks
    @test true  # placeholder to keep the test runner happy
end
```

- [ ] **Step 5: Create root `.gitignore`**

```
*.jl.cov
*.jl.*.cov
*.jl.mem
deps/deps.jl
deps/build.log

# Manifest files — committed at the lectures level, but skip for now
code/Manifest.toml
docs/Manifest.toml

# Generated artifacts
scripts/data/*.jld2
!scripts/data/.gitkeep
!scripts/data/frozen_basket.jld2

# Notebook outputs
.ipynb_checkpoints/

# OS junk
.DS_Store

# Editor
.vscode/
```

- [ ] **Step 6: Instantiate dependencies and verify the package loads**

Run:
```bash
cd /Users/jdv27/Desktop/julia_work/modified_cobb_douglas_example
julia --project=code -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
julia --project=code -e 'using ConstrainedCobbDouglas; println("OK")'
```

Expected output: `OK` on the last line (after potentially a few minutes of compile time).

- [ ] **Step 7: Run the placeholder test**

Run:
```bash
julia --project=code -e 'using Pkg; Pkg.test()'
```

Expected: 1 test passing.

- [ ] **Step 8: Commit**

```bash
cd /Users/jdv27/Desktop/julia_work/modified_cobb_douglas_example
mkdir -p scripts/data
touch scripts/data/.gitkeep
git add code/Project.toml code/src/ConstrainedCobbDouglas.jl code/src/Types.jl \
        code/test/runtests.jl .gitignore scripts/data/.gitkeep
git commit -m "Initialize ConstrainedCobbDouglas package skeleton"
```

---

### Task 2: Copy committed data files from lectures repo

**Files:**
- Create: `code/src/data/SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2`
- Create: `code/src/data/SP500-Daily-OHLC-1-2-2025-to-12-31-2025.jld2`
- Create: `code/src/data/SP500-Daily-OHLC-1-2-2026-to-04-22-2026.jld2`
- Create: `code/src/data/sp500-sectors.csv`
- Create: `code/src/data/pretrained-jumphmm-market-surrogate.jld2`

- [ ] **Step 1: Copy data files from lectures repo**

```bash
SRC=/Users/jdv27/Desktop/julia_work/eCornell-AI-finance-lectures/code/src/data
DST=/Users/jdv27/Desktop/julia_work/modified_cobb_douglas_example/code/src/data
mkdir -p $DST
cp $SRC/SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2 $DST/
cp $SRC/SP500-Daily-OHLC-1-2-2025-to-12-31-2025.jld2 $DST/
cp $SRC/SP500-Daily-OHLC-1-2-2026-to-04-22-2026.jld2 $DST/
cp $SRC/sp500-sectors.csv $DST/
cp $SRC/pretrained-jumphmm-market-surrogate.jld2 $DST/
```

- [ ] **Step 2: Verify files copied correctly**

Run:
```bash
ls -la code/src/data/
```

Expected: 5 files (3 .jld2 OHLC, 1 .csv sectors, 1 .jld2 jumphmm).

- [ ] **Step 3: Verify JLD2 files load**

Run:
```bash
julia --project=code -e 'using JLD2; d = load("code/src/data/SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2"); println("keys: ", keys(d))'
```

Expected: prints a list of keys (e.g., "dataset", "tickers", "dates" — exact keys depend on lectures format).

- [ ] **Step 4: Commit**

```bash
git add code/src/data/
git commit -m "Vendor committed data inputs from lectures repo (OHLC, sectors, JumpHMM surrogate)"
```

---

## Phase 1 — Types

### Task 3: Create Types.jl with all struct definitions

**Files:**
- Modify: `code/src/Types.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl` (export types)
- Create: `code/test/test_types.jl`
- Modify: `code/test/runtests.jl` (include the new test file)

- [ ] **Step 1: Write the failing test** for type construction

Create `code/test/test_types.jl`:

```julia
using Test
using Dates
using ConstrainedCobbDouglas

@testset "Type construction" begin
    @testset "MySIMParameterEstimate" begin
        est = MySIMParameterEstimate()
        est.ticker = "AAPL"; est.α = 0.05; est.β = 1.2; est.σ_ε = 0.20; est.r² = 0.65
        @test est.ticker == "AAPL"
        @test est.α == 0.05
    end

    @testset "MyEWLSState" begin
        s = MyEWLSState()
        s.α = 0.0; s.β = 1.0; s.σ_ε = 0.2
        s.Sw = 1.0; s.Swx = 0.0; s.Swy = 0.0
        s.Swxx = 1.0; s.Swxy = 1.0; s.Swyy = 1.0; s.η = 0.99
        @test s.β == 1.0
    end

    @testset "MyConstrainedCDProblem and Result" begin
        p = MyConstrainedCDProblem(
            γ = [0.5, 0.3], p = [100.0, 50.0], B = 10000.0,
            Σ = [0.04 0.01; 0.01 0.09], σ_max = 0.15,
            K_turnover = 1000.0, w_max = 0.20,
            n_prev = [10.0, 20.0], c̄ = 0.05)
        @test p.B == 10000.0
        @test length(p.γ) == 2

        r = MyConstrainedCDResult(
            n = [50.0, 100.0], w = [0.5, 0.5],
            unallocated_budget = 0.0,
            duals = (σ_max = 0.0, turnover = 0.0, w_max = 0.0),
            status = :optimal,
            objective = 1.234)
        @test r.status == :optimal
    end

    @testset "MyMPCSpec, MyMPCProjection, MyMPCTrigger" begin
        spec = MyMPCSpec(z = 1.96, T = 21, N = 1000, D_max = 0.08)
        @test spec.T == 21

        proj = MyMPCProjection(
            μ = ones(21), σ = 0.01 * ones(21),
            V₀ = 100_000.0, paths = ones(1000, 21),
            decision_date_idx = 1,
            closed_form_μ = ones(21), closed_form_σ = 0.01 * ones(21),
            divergence_warning = false)
        @test length(proj.μ) == 21

        trig = MyMPCTrigger(fired = false, reason = :in_spec, τ = 5)
        @test trig.reason == :in_spec
    end

    @testset "MyCostModel" begin
        cm = MyCostModel(
            commission_per_trade = 0.0,
            half_spread_bps = 5.0,
            slippage_κ = 0.001,
            adv = Dict("AAPL" => 1.0e7))
        @test cm.half_spread_bps == 5.0
    end

    @testset "MyTaxLot and MyTaxLedger" begin
        lot = MyTaxLot(ticker = "AAPL", open_date = Date(2025,1,2),
                       open_price = 150.0, qty = 100)
        @test lot.qty == 100

        ledger = MyTaxLedger()
        @test isempty(ledger.lots)
        @test ledger.realized_st_pnl == 0.0
    end

    @testset "Strategy types" begin
        s1 = EqualWeightStrategy()
        s2 = MinVarBuyHoldStrategy()
        s3 = UnconstrainedCDStrategy()
        s4 = CostAwareMVStrategy(κ = 5.0, c = 0.0005)
        spec = MyMPCSpec(z = 1.96, T = 21, N = 1000, D_max = 0.08)
        s5 = CDWithMPCStrategy(spec = spec)
        s6 = ConstrainedCDWithMPCStrategy(spec = spec, σ_max = 0.12,
                                         K_turnover = 10_000.0, w_max = 0.20)
        @test isa(s1, MyAllocationStrategy)
        @test isa(s6, MyAllocationStrategy)
        @test s4.κ == 5.0
        @test s6.σ_max == 0.12
    end
end
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
julia --project=code -e 'using Pkg; Pkg.activate("code"); include("code/test/test_types.jl")'
```

Expected: FAIL (types not defined).

- [ ] **Step 3: Implement `code/src/Types.jl`**

```julia
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
```

- [ ] **Step 4: Export types from `code/src/ConstrainedCobbDouglas.jl`**

Add this block after the `include("Types.jl")` line in `code/src/ConstrainedCobbDouglas.jl`:

```julia
export MySIMParameterEstimate, MyEWLSState
export MyConstrainedCDProblem, MyConstrainedCDResult
export MyMPCSpec, MyMPCProjection, MyMPCTrigger
export MyCostModel
export MyTaxLot, MyTaxLedger
export MyAllocationStrategy, EqualWeightStrategy, MinVarBuyHoldStrategy,
       UnconstrainedCDStrategy, CostAwareMVStrategy,
       CDWithMPCStrategy, ConstrainedCDWithMPCStrategy
export MyBanditConfig, MyBanditResult
export MyBacktestState, MyBacktestResult
```

- [ ] **Step 5: Wire the new test into `code/test/runtests.jl`**

Replace contents with:

```julia
using Test
using ConstrainedCobbDouglas

@testset "ConstrainedCobbDouglas.jl" begin
    include("test_types.jl")
end
```

- [ ] **Step 6: Run the test**

```bash
julia --project=code -e 'using Pkg; Pkg.test()'
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add code/src/Types.jl code/src/ConstrainedCobbDouglas.jl \
        code/test/test_types.jl code/test/runtests.jl
git commit -m "Add Types.jl with all struct definitions for the package"
```

---

## Phase 2 — SIM module

### Task 4: Vendor `estimate_sim` and `build_sim_covariance`

**Files:**
- Create: `code/src/SIM.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl` (include + export)
- Create: `code/test/test_sim.jl`
- Modify: `code/test/runtests.jl`

- [ ] **Step 1: Write the failing tests**

Create `code/test/test_sim.jl`:

```julia
using Test
using LinearAlgebra
using Random
using Statistics
using ConstrainedCobbDouglas

@testset "SIM module" begin
    @testset "estimate_sim recovers known parameters" begin
        Random.seed!(42)
        T = 2000
        α_true, β_true, σ_ε_true = 0.03, 1.1, 0.18
        g_m = 0.08 .+ 0.15 .* randn(T)
        ε = σ_ε_true .* randn(T)
        g_i = α_true .+ β_true .* g_m .+ ε
        est = estimate_sim(g_m, g_i, "TEST")
        @test isapprox(est.α, α_true; atol = 0.02)
        @test isapprox(est.β, β_true; atol = 0.02)
        @test isapprox(est.σ_ε, σ_ε_true; atol = 0.02)
        @test est.r² > 0.5
        @test est.ticker == "TEST"
    end

    @testset "build_sim_covariance is symmetric PSD" begin
        ests = MySIMParameterEstimate[]
        for (t, β, σ_ε) in [("A", 1.0, 0.2), ("B", 0.8, 0.15), ("C", 1.3, 0.25)]
            e = MySIMParameterEstimate()
            e.ticker = t; e.α = 0.05; e.β = β; e.σ_ε = σ_ε; e.r² = 0.5
            push!(ests, e)
        end
        Σ = build_sim_covariance(ests, 0.15)
        @test size(Σ) == (3, 3)
        @test Σ ≈ Σ'                      # symmetric
        @test all(eigvals(Σ) .> -1e-10)   # PSD
        @test Σ[1, 1] ≈ 1.0^2 * 0.15^2 + 0.2^2
        @test Σ[1, 2] ≈ 1.0 * 0.8 * 0.15^2
    end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
julia --project=code -e 'using Pkg; Pkg.activate("code"); include("code/test/test_sim.jl")'
```

Expected: FAIL (`estimate_sim` / `build_sim_covariance` not defined).

- [ ] **Step 3: Create `code/src/SIM.jl`** with vendored functions

```julia
# --- estimate_sim and build_sim_covariance vendored from
# eCornell-AI-finance-lectures/code/src/Compute.jl (lines 96-175). ---

"""
    estimate_sim(market_returns, asset_returns, ticker; δ = 0.0) -> MySIMParameterEstimate

Fit α, β, σ_ε via regularized OLS on annualized growth rates.
Inputs and outputs are annualized (1/year units).
"""
function estimate_sim(market_returns::Array{Float64,1}, asset_returns::Array{Float64,1},
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
function build_sim_covariance(sim_estimates::Array{MySIMParameterEstimate,1},
        σ_m::Float64)::Array{Float64,2}
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
```

- [ ] **Step 4: Include and export from `code/src/ConstrainedCobbDouglas.jl`**

Uncomment `include("SIM.jl")` in the module file and add to the exports block:

```julia
export estimate_sim, build_sim_covariance
```

- [ ] **Step 5: Wire into `code/test/runtests.jl`**

Add `include("test_sim.jl")` inside the outer `@testset`.

- [ ] **Step 6: Run tests**

```bash
julia --project=code -e 'using Pkg; Pkg.test()'
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add code/src/SIM.jl code/src/ConstrainedCobbDouglas.jl \
        code/test/test_sim.jl code/test/runtests.jl
git commit -m "SIM.jl: estimate_sim + build_sim_covariance (vendored from lectures)"
```

---

### Task 5: Add market growth, EMA, λ, preference-weights functions

**Files:**
- Modify: `code/src/SIM.jl`
- Modify: `code/test/test_sim.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl` (exports)

- [ ] **Step 1: Append failing tests to `code/test/test_sim.jl`**

Add these `@testset` blocks inside the outer `@testset "SIM module"`:

```julia
@testset "compute_market_growth annualizes log-returns" begin
    prices = [100.0, 101.0, 100.5, 102.0]
    g = compute_market_growth(prices; Δt = 1.0 / 252.0)
    @test length(g) == 3
    @test g[1] ≈ log(101.0 / 100.0) / (1.0 / 252.0)
end

@testset "compute_ema length and bounds" begin
    prices = collect(100.0:1.0:300.0)
    ema = compute_ema(prices; window = 21)
    @test length(ema) == length(prices)
    @test ema[end] > prices[1]
    @test ema[end] < prices[end]
end

@testset "compute_lambda is bounded in [0,1]" begin
    prices = collect(100.0:1.0:300.0)
    short_ema = compute_ema(prices; window = 21)
    long_ema  = compute_ema(prices; window = 63)
    λ = compute_lambda(short_ema, long_ema)
    @test all(0.0 .<= λ .<= 1.0)
end

@testset "compute_preference_weights returns tanh-bounded γ" begin
    sim_params = Dict("A" => (0.05, 1.0, 0.2), "B" => (0.10, 1.5, 0.3))
    tickers = ["A", "B"]
    γ = compute_preference_weights(sim_params, tickers, 0.08, 0.5)
    @test length(γ) == 2
    @test all(-1.0 .<= γ .<= 1.0)
end
```

- [ ] **Step 2: Run tests; expect failure**

```bash
julia --project=code -e 'using Pkg; Pkg.test()'
```

Expected: FAIL (new functions not defined).

- [ ] **Step 3: Append implementations to `code/src/SIM.jl`**

```julia
# --- compute_market_growth, compute_ema, compute_lambda vendored
# from Compute.jl lines 360-460. ---

"""
    compute_market_growth(prices; Δt = 1/252) -> Vector{Float64}

Annualized log-returns of a price series.
"""
function compute_market_growth(prices::Array{Float64,1};
        Δt::Float64 = 1.0 / 252.0)::Array{Float64,1}
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
function compute_ema(prices::Array{Float64,1}; window::Int = 21)::Array{Float64,1}
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
function compute_lambda(short_ema::Array{Float64,1}, long_ema::Array{Float64,1};
        θ::Float64 = 0.5)::Array{Float64,1}
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
        tickers::Array{String,1}, gm_t::Float64, lambda::Float64)::Array{Float64,1}
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
```

- [ ] **Step 4: Export from module**

Add to the export block in `code/src/ConstrainedCobbDouglas.jl`:

```julia
export compute_market_growth, compute_ema, compute_lambda, compute_preference_weights
```

- [ ] **Step 5: Run tests**

```bash
julia --project=code -e 'using Pkg; Pkg.test()'
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add code/src/SIM.jl code/src/ConstrainedCobbDouglas.jl code/test/test_sim.jl
git commit -m "SIM.jl: market growth, EMA, lambda, preference weights (no-news)"
```

---

### Task 6: Add `ewls_init` and `ewls_update!`

**Files:**
- Modify: `code/src/SIM.jl`
- Modify: `code/test/test_sim.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl` (exports)

- [ ] **Step 1: Append failing tests**

Add inside `@testset "SIM module"`:

```julia
@testset "ewls_init recovers prior estimates" begin
    s = ewls_init(0.05, 1.2, 0.18; half_life = 252.0, prior_weight = 252.0)
    @test isapprox(s.α, 0.05; atol = 1e-12)
    @test isapprox(s.β, 1.2; atol = 1e-12)
    @test isapprox(s.σ_ε, 0.18; atol = 1e-6)
    @test s.η > 0.99 && s.η < 1.0
end

@testset "ewls_update! tracks a step change after enough data" begin
    Random.seed!(123)
    s = ewls_init(0.0, 1.0, 0.2; half_life = 21.0, prior_weight = 21.0)
    α_new, β_new, σ_new = 0.10, 1.5, 0.20
    for _ in 1:500
        g_m = 0.15 * randn()
        g_i = α_new + β_new * g_m + σ_new * randn()
        ewls_update!(s, g_i, g_m)
    end
    @test isapprox(s.β, β_new; atol = 0.10)
    @test isapprox(s.α, α_new; atol = 0.05)
end

@testset "ewls_update! decay factor" begin
    s = ewls_init(0.0, 1.0, 0.1; half_life = 252.0, prior_weight = 252.0)
    expected_η = 2.0^(-1.0 / 252.0)
    @test isapprox(s.η, expected_η; atol = 1e-12)
end
```

- [ ] **Step 2: Run tests; expect failure**

```bash
julia --project=code -e 'using Pkg; Pkg.test()'
```

Expected: FAIL.

- [ ] **Step 3: Append to `code/src/SIM.jl`**

```julia
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
```

- [ ] **Step 4: Export from module**

Add to the export block:

```julia
export ewls_init, ewls_update!
```

- [ ] **Step 5: Run tests**

```bash
julia --project=code -e 'using Pkg; Pkg.test()'
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add code/src/SIM.jl code/src/ConstrainedCobbDouglas.jl code/test/test_sim.jl
git commit -m "SIM.jl: ewls_init + ewls_update! for online parameter tracking"
```

---

## Phase 3 — Allocator module (subset; remaining tasks in plan-pt2)

### Task 7: Add analytical Cobb-Douglas, equal-weight, min-variance allocators

**Files:**
- Create: `code/src/Allocator.jl`
- Create: `code/test/test_allocator.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl`
- Modify: `code/test/runtests.jl`

- [ ] **Step 1: Write failing tests**

Create `code/test/test_allocator.jl`:

```julia
using Test
using LinearAlgebra
using JuMP
using Clarabel
using ConstrainedCobbDouglas

@testset "Allocator: analytical and baselines" begin
    @testset "solve_unconstrained_cd_analytical: budget identity" begin
        γ = [0.5, 0.3, 0.2]
        p = [100.0, 50.0, 25.0]
        B = 10_000.0
        n, cash = solve_unconstrained_cd_analytical(γ, p, B; ε = 1e-4)
        @test sum(n .* p) + cash ≈ B atol=1e-6
        @test n[1] / n[2] ≈ (γ[1] / γ[2]) * (p[2] / p[1]) atol=1e-6
    end

    @testset "solve_unconstrained_cd_analytical: all γ ≤ 0 returns cash" begin
        γ = [-0.1, -0.05]
        p = [100.0, 50.0]
        B = 10_000.0
        n, cash = solve_unconstrained_cd_analytical(γ, p, B; ε = 1e-4)
        @test all(n .≈ 1e-4)
        @test cash ≈ B - sum(n .* p) atol=1e-6
    end

    @testset "equal_weight_target sums to 1.0" begin
        w = equal_weight_target(5)
        @test sum(w) ≈ 1.0
        @test all(w .≈ 0.2)
    end

    @testset "solve_minvar_buyhold returns weights on a feasible problem" begin
        Σ = [0.04 0.01 0.0; 0.01 0.09 0.02; 0.0 0.02 0.16]
        bounds = [0.0 1.0; 0.0 1.0; 0.0 1.0]
        w = solve_minvar_buyhold(Σ, bounds)
        @test length(w) == 3
        @test sum(w) ≈ 1.0 atol=1e-6
        @test all(w .≥ -1e-8)
    end
end
```

- [ ] **Step 2: Run; expect failure**

```bash
julia --project=code -e 'using Pkg; Pkg.test()'
```

Expected: FAIL.

- [ ] **Step 3: Create `code/src/Allocator.jl`**

```julia
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
```

- [ ] **Step 4: Include and export from module**

Uncomment `include("Allocator.jl")` and add:

```julia
export solve_unconstrained_cd_analytical, equal_weight_target, solve_minvar_buyhold
```

- [ ] **Step 5: Wire test file into runtests.jl** (add `include("test_allocator.jl")`)

- [ ] **Step 6: Run tests**

```bash
julia --project=code -e 'using Pkg; Pkg.test()'
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add code/src/Allocator.jl code/src/ConstrainedCobbDouglas.jl \
        code/test/test_allocator.jl code/test/runtests.jl
git commit -m "Allocator.jl: analytical unconstrained CD + EW + min-variance baselines"
```

---

### Task 8: Implement the constrained CD JuMP solver — shell with budget + log objective

**Files:**
- Modify: `code/src/Allocator.jl`
- Modify: `code/test/test_allocator.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl` (exports)

- [ ] **Step 1: Add failing test for the loose-constraint identity**

Append to `code/test/test_allocator.jl` inside the existing `@testset`:

```julia
@testset "solve_constrained_cd: loose-constraint identity vs analytical" begin
    γ = [0.5, 0.3, 0.2]
    p = [100.0, 50.0, 25.0]
    B = 10_000.0
    Σ = Matrix{Float64}(I, 3, 3) * 0.04
    problem = MyConstrainedCDProblem(
        γ = γ, p = p, B = B, Σ = Σ,
        σ_max = 1.0e6,            # effectively infinite
        K_turnover = 1.0e12,      # effectively infinite
        w_max = 1.0,
        n_prev = zeros(3),
        c̄ = 0.0)
    res = solve_constrained_cd(problem)
    n_an, _ = solve_unconstrained_cd_analytical(γ, p, B; ε = 1e-3)
    @test res.status == :optimal
    @test sum(res.n .* p) ≤ B + 1e-3
    # ratio test (numerical comparison, since both solve max Σ γ log n)
    for i in 1:3, j in 1:3
        if n_an[i] > 1e-6 && n_an[j] > 1e-6
            @test isapprox(res.n[i] / res.n[j], n_an[i] / n_an[j]; rtol = 5e-3)
        end
    end
end
```

- [ ] **Step 2: Run; expect failure** (`solve_constrained_cd` not defined).

```bash
julia --project=code -e 'using Pkg; Pkg.test()'
```

- [ ] **Step 3: Append `solve_constrained_cd` to `code/src/Allocator.jl`**

```julia
# --- Constrained Cobb-Douglas solver (spec §4) ---

"""
    solve_constrained_cd(problem::MyConstrainedCDProblem) -> MyConstrainedCDResult

Solve max Σ γᵢ log(nᵢ) s.t. budget, covariance, turnover, and concentration
constraints (spec §4). Uses Clarabel via JuMP; falls back to SCS if Clarabel
returns a non-`OPTIMAL` status.

Non-preferred assets (γᵢ ≤ 0) are pinned at ε = 1e-3 shares; the optimizer
runs over the preferred subset only (keeps the objective concave).
"""
function solve_constrained_cd(problem::MyConstrainedCDProblem;
        ε::Float64 = 1e-3)::MyConstrainedCDResult
    γ = problem.γ; p = problem.p; B = problem.B
    Σ = problem.Σ; σ_max = problem.σ_max
    K_turnover = problem.K_turnover; w_max = problem.w_max
    n_prev = problem.n_prev; c̄ = problem.c̄
    K = length(γ)
    preferred = findall(γ .> 0)
    non_pref = findall(γ .<= 0)

    if isempty(preferred)
        n_full = zeros(K)
        for i in non_pref; n_full[i] = ε; end
        return MyConstrainedCDResult(
            n = n_full,
            w = (n_full .* p) ./ B,
            unallocated_budget = B - sum(n_full .* p),
            duals = (σ_max = 0.0, turnover = 0.0, w_max = 0.0),
            status = :no_preferred,
            objective = 0.0)
    end

    # Regularize Σ for Cholesky
    Σ_reg = Σ + 1e-8 * I(K)
    L = cholesky(Σ_reg).L

    pinned_cost = sum(ε * p[i] for i in non_pref; init = 0.0)
    B_eff = B - pinned_cost

    Kp = length(preferred)
    p_p = p[preferred]
    n_prev_p = n_prev[preferred]

    function build_model(opt)
        m = Model(opt)
        set_silent(m)
        @variable(m, n[1:Kp] >= 1e-8)
        @variable(m, t[1:Kp])
        # log epigraph via exponential cone: t ≤ log(n)  ⇔  (t, 1, n) ∈ ExpCone
        for k in 1:Kp
            @constraint(m, [t[k], 1.0, n[k]] in MOI.ExponentialCone())
        end
        @objective(m, Max, sum(γ[preferred[k]] * t[k] for k in 1:Kp))

        # Budget
        @constraint(m, sum(n[k] * p_p[k] for k in 1:Kp) <= B_eff)

        # Full weight vector (preferred + pinned non-preferred)
        @expression(m, w_full[i = 1:K],
            (i in preferred) ?
                n[findfirst(==(i), preferred)] * p[i] / B :
                ε * p[i] / B)

        # Concentration cap
        for i in 1:K
            @constraint(m, w_full[i] <= w_max)
        end

        # Covariance budget via SOC: ||Lᵀ w|| ≤ σ_max
        Lt = Matrix(L')
        @constraint(m, [σ_max; Lt * collect(w_full)] in SecondOrderCone())

        # Turnover budget (l1) — slack vars on preferred only; non-preferred churn ignored
        # since their position changes ε → ε (zero) under the same regime.
        @variable(m, u[1:Kp] >= 0)
        for k in 1:Kp
            @constraint(m, u[k] >= n[k] - n_prev_p[k])
            @constraint(m, u[k] >= n_prev_p[k] - n[k])
        end
        @constraint(m, c̄ * sum(u) <= K_turnover)
        return m, n, t
    end

    # Try Clarabel first
    m, nvar, tvar = build_model(Clarabel.Optimizer)
    optimize!(m)
    status = termination_status(m)

    if status != MOI.OPTIMAL
        # Fall back to SCS
        m, nvar, tvar = build_model(SCS.Optimizer)
        optimize!(m)
        status = termination_status(m)
    end

    if status != MOI.OPTIMAL
        return MyConstrainedCDResult(
            n = copy(n_prev),
            w = (n_prev .* p) ./ B,
            unallocated_budget = 0.0,
            duals = (σ_max = 0.0, turnover = 0.0, w_max = 0.0),
            status = :solver_failed,
            objective = 0.0)
    end

    n_p = value.(nvar)
    n_full = zeros(K)
    for (k, i) in enumerate(preferred); n_full[i] = n_p[k]; end
    for i in non_pref; n_full[i] = ε; end
    obj = objective_value(m)
    w_full = (n_full .* p) ./ B
    return MyConstrainedCDResult(
        n = n_full,
        w = w_full,
        unallocated_budget = max(0.0, B - sum(n_full .* p)),
        duals = (σ_max = 0.0, turnover = 0.0, w_max = 0.0),  # filled in Task 10
        status = :optimal,
        objective = obj)
end
```

- [ ] **Step 4: Export `solve_constrained_cd`**

Add to the export block in `code/src/ConstrainedCobbDouglas.jl`:

```julia
export solve_constrained_cd
```

- [ ] **Step 5: Run tests**

```bash
julia --project=code -e 'using Pkg; Pkg.test()'
```

Expected: pass (Clarabel may take a moment to compile on first run).

- [ ] **Step 6: Commit**

```bash
git add code/src/Allocator.jl code/src/ConstrainedCobbDouglas.jl code/test/test_allocator.jl
git commit -m "Allocator.jl: JuMP/Clarabel constrained CD solver (loose-constraint identity)"
```

---

### Task 9: Constrained-CD edge cases — σ_max monotonicity, zero-turnover lock, concentration, no-preferred

**Files:**
- Modify: `code/test/test_allocator.jl`

- [ ] **Step 1: Append edge-case tests**

```julia
@testset "solve_constrained_cd: σ_max monotonicity" begin
    γ = [0.5, 0.3, 0.2]
    p = [100.0, 50.0, 25.0]
    B = 10_000.0
    βs = [1.0, 0.8, 1.3]
    σ_m = 0.15
    σ_εs = [0.20, 0.15, 0.25]
    Σ = zeros(3, 3)
    for i in 1:3, j in 1:3
        Σ[i, j] = (i == j) ? βs[i]^2 * σ_m^2 + σ_εs[i]^2 : βs[i] * βs[j] * σ_m^2
    end
    function port_var(σ_max_val)
        problem = MyConstrainedCDProblem(
            γ = γ, p = p, B = B, Σ = Σ,
            σ_max = σ_max_val, K_turnover = 1e12, w_max = 1.0,
            n_prev = zeros(3), c̄ = 0.0)
        r = solve_constrained_cd(problem)
        return dot(r.w, Σ * r.w)
    end
    v_loose = port_var(0.50)
    v_mid   = port_var(0.20)
    v_tight = port_var(0.08)
    @test v_loose >= v_mid - 1e-8
    @test v_mid   >= v_tight - 1e-8
end

@testset "solve_constrained_cd: zero-turnover lock" begin
    γ = [0.5, 0.3, 0.2]
    p = [100.0, 50.0, 25.0]
    B = 10_000.0
    n_prev, _ = solve_unconstrained_cd_analytical(γ, p, B; ε = 1e-3)
    problem = MyConstrainedCDProblem(
        γ = γ, p = p, B = B, Σ = Matrix{Float64}(I, 3, 3) * 0.04,
        σ_max = 1e6, K_turnover = 0.0, w_max = 1.0,
        n_prev = n_prev, c̄ = 0.05)
    r = solve_constrained_cd(problem)
    @test r.status == :optimal
    @test all(isapprox.(r.n, n_prev; atol = 1e-4))
end

@testset "solve_constrained_cd: concentration cap binds" begin
    γ = [0.99, 0.005, 0.005]
    p = [100.0, 50.0, 25.0]
    B = 10_000.0
    problem = MyConstrainedCDProblem(
        γ = γ, p = p, B = B, Σ = Matrix{Float64}(I, 3, 3) * 0.04,
        σ_max = 1e6, K_turnover = 1e12, w_max = 0.40,
        n_prev = zeros(3), c̄ = 0.0)
    r = solve_constrained_cd(problem)
    @test r.status == :optimal
    @test maximum(r.w) <= 0.40 + 1e-3
end

@testset "solve_constrained_cd: no-preferred fallback" begin
    γ = [-0.1, -0.05, -0.2]
    p = [100.0, 50.0, 25.0]
    B = 10_000.0
    problem = MyConstrainedCDProblem(
        γ = γ, p = p, B = B, Σ = Matrix{Float64}(I, 3, 3) * 0.04,
        σ_max = 1e6, K_turnover = 1e12, w_max = 1.0,
        n_prev = zeros(3), c̄ = 0.0)
    r = solve_constrained_cd(problem)
    @test r.status == :no_preferred
    @test r.unallocated_budget > 0.0
end
```

- [ ] **Step 2: Run tests**

```bash
julia --project=code -e 'using Pkg; Pkg.test()'
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add code/test/test_allocator.jl
git commit -m "Allocator.jl tests: σ_max monotonicity, zero-turnover lock, concentration, no-preferred"
```

---

### Task 10: Implement `solve_cost_aware_mv` (strategy 4)

**Files:**
- Modify: `code/src/Allocator.jl`
- Modify: `code/test/test_allocator.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl` (exports)

- [ ] **Step 1: Append failing test**

```julia
@testset "solve_cost_aware_mv produces a feasible weight vector" begin
    γ = [0.4, 0.3, 0.2, 0.1]
    Σ = Matrix{Float64}(I, 4, 4) * 0.04
    w_prev = [0.25, 0.25, 0.25, 0.25]
    w = solve_cost_aware_mv(γ, Σ, w_prev; κ = 5.0, c = 0.001)
    @test length(w) == 4
    @test sum(w) ≈ 1.0 atol=1e-6
    @test all(w .>= -1e-8)
end
```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Append to `code/src/Allocator.jl`**

```julia
"""
    solve_cost_aware_mv(γ, Σ, w_prev; κ, c) -> Vector{Float64}

Strategy 4: max γᵀw - (κ/2) wᵀΣw - c·‖w - w_prev‖₁  s.t. Σwᵢ = 1, wᵢ ≥ 0.
"""
function solve_cost_aware_mv(γ::Vector{Float64}, Σ::Matrix{Float64},
        w_prev::Vector{Float64}; κ::Float64, c::Float64)::Vector{Float64}
    K = length(γ)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, w[1:K] >= 0)
    @variable(model, u[1:K] >= 0)
    @constraint(model, sum(w) == 1.0)
    for i in 1:K
        @constraint(model, u[i] >=  w[i] - w_prev[i])
        @constraint(model, u[i] >= -(w[i] - w_prev[i]))
    end
    @objective(model, Max, sum(γ[i] * w[i] for i in 1:K)
                          - (κ / 2.0) * w' * Σ * w
                          - c * sum(u))
    optimize!(model)
    return value.(w)
end
```

- [ ] **Step 4: Export**

```julia
export solve_cost_aware_mv
```

- [ ] **Step 5: Run tests; expect pass**

- [ ] **Step 6: Commit**

```bash
git add code/src/Allocator.jl code/src/ConstrainedCobbDouglas.jl code/test/test_allocator.jl
git commit -m "Allocator.jl: cost-aware MV solver (strategy 4)"
```

---

## Phase 4 — Cost engine

### Task 11: Implement `MyCostModel` constructor + `trade_cost`

**Files:**
- Create: `code/src/Costs.jl`
- Create: `code/test/test_costs.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl`
- Modify: `code/test/runtests.jl`

- [ ] **Step 1: Write failing tests**

Create `code/test/test_costs.jl`:

```julia
using Test
using ConstrainedCobbDouglas

@testset "Costs module" begin
    cm = MyCostModel(
        commission_per_trade = 0.0,
        half_spread_bps = 5.0,
        slippage_κ = 0.001,
        adv = Dict("AAPL" => 1.0e8))

    @testset "Round-trip half-spread cost" begin
        c_buy  = trade_cost(cm, "AAPL", +100, 100.0)
        c_sell = trade_cost(cm, "AAPL", -100, 100.0)
        # Half-spread on each leg: 5e-4 * 100 * 100 = $5; round-trip = $10
        @test c_buy + c_sell ≈ 10.0 atol=1e-6
    end

    @testset "Slippage scales quadratically with order size" begin
        cm_no_spread = MyCostModel(
            commission_per_trade = 0.0,
            half_spread_bps = 0.0,
            slippage_κ = 0.001,
            adv = Dict("X" => 1.0e6))
        c_100   = trade_cost(cm_no_spread, "X", 100,   50.0)
        c_1000  = trade_cost(cm_no_spread, "X", 1000,  50.0)
        @test c_1000 / c_100 ≈ 100.0 atol=1e-3
    end

    @testset "Zero-share order is zero cost" begin
        @test trade_cost(cm, "AAPL", 0, 100.0) ≈ 0.0
    end

    @testset "Commission is flat" begin
        cm_with_commish = MyCostModel(
            commission_per_trade = 1.0,
            half_spread_bps = 0.0,
            slippage_κ = 0.0,
            adv = Dict("X" => 1.0e9))
        @test trade_cost(cm_with_commish, "X", 1, 10.0) ≈ 1.0
        @test trade_cost(cm_with_commish, "X", 100, 10.0) ≈ 1.0
    end
end
```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Create `code/src/Costs.jl`**

```julia
"""
    trade_cost(model::MyCostModel, ticker, q_signed, price) -> Float64

Per-fill cost: half-spread + linear-impact slippage + flat commission.
`q_signed` positive for buy, negative for sell.
"""
function trade_cost(model::MyCostModel, ticker::String, q_signed::Int,
        price::Float64)::Float64
    if q_signed == 0
        return 0.0
    end
    q = abs(q_signed)
    adv_t = get(model.adv, ticker, 1.0e9)   # default huge ADV → ~no slippage
    half_spread_cost = (model.half_spread_bps * 1e-4) * price * q
    slippage_cost    = model.slippage_κ * (q / adv_t) * price * q
    return half_spread_cost + slippage_cost + model.commission_per_trade
end
```

- [ ] **Step 4: Include and export from module**

Uncomment `include("Costs.jl")` and add:

```julia
export trade_cost
```

- [ ] **Step 5: Wire into `code/test/runtests.jl`** (add `include("test_costs.jl")`)

- [ ] **Step 6: Run tests; expect pass**

- [ ] **Step 7: Commit**

```bash
git add code/src/Costs.jl code/src/ConstrainedCobbDouglas.jl \
        code/test/test_costs.jl code/test/runtests.jl
git commit -m "Costs.jl: trade_cost (half-spread + linear-impact slippage + commission)"
```

---

## Phase 5 — Tax engine

### Task 12: Implement `MyTaxLedger` + `open_lot!` + `close_qty!` with FIFO

**Files:**
- Create: `code/src/Tax.jl`
- Create: `code/test/test_tax.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl`
- Modify: `code/test/runtests.jl`

- [ ] **Step 1: Write failing tests**

Create `code/test/test_tax.jl`:

```julia
using Test
using Dates
using ConstrainedCobbDouglas

@testset "Tax module" begin
    @testset "FIFO ordering — closes the older lot first" begin
        led = MyTaxLedger()
        open_lot!(led, "AAPL", 100, 50.0, Date(2025,1,2))
        open_lot!(led, "AAPL", 100, 60.0, Date(2025,2,1))
        close_qty!(led, "AAPL", 50, 70.0, Date(2025,3,1))
        @test length(led.lots["AAPL"]) == 2
        @test led.lots["AAPL"][1].qty == 50   # remainder of the older lot
        @test led.realized_st_pnl ≈ 50 * (70.0 - 50.0)
        @test led.realized_lt_pnl == 0.0
        @test length(led.closed_lots) == 1
    end

    @testset "Partial close shrinks front lot" begin
        led = MyTaxLedger()
        open_lot!(led, "X", 100, 50.0, Date(2025,1,2))
        close_qty!(led, "X", 30, 60.0, Date(2025,2,1))
        @test led.lots["X"][1].qty == 70
        @test led.realized_st_pnl ≈ 30 * 10.0
    end

    @testset "ST/LT boundary at 365 days" begin
        led = MyTaxLedger()
        open_lot!(led, "X", 100, 50.0, Date(2025,1,2))
        close_qty!(led, "X", 50, 60.0, Date(2025,1,2) + Day(364))
        close_qty!(led, "X", 50, 60.0, Date(2025,1,2) + Day(365))
        @test led.realized_st_pnl ≈ 50 * 10.0
        @test led.realized_lt_pnl ≈ 50 * 10.0
    end

    @testset "Over-close throws" begin
        led = MyTaxLedger()
        open_lot!(led, "X", 50, 50.0, Date(2025,1,2))
        @test_throws ErrorException close_qty!(led, "X", 60, 60.0, Date(2025,2,1))
    end
end
```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Create `code/src/Tax.jl`**

```julia
"""
    open_lot!(ledger, ticker, qty, price, date)

Append a new tax lot to the back of the FIFO queue for `ticker`.
"""
function open_lot!(ledger::MyTaxLedger, ticker::String, qty::Int,
        price::Float64, date::Date)
    queue = get!(ledger.lots, ticker, MyTaxLot[])
    push!(queue, MyTaxLot(ticker = ticker, open_date = date,
                          open_price = price, qty = qty))
    return nothing
end

"""
    close_qty!(ledger, ticker, qty_to_close, price, date)

Consume the FIFO queue from the front, accumulating ST/LT P&L based on
the 365-day holding-period boundary. Throws if the requested qty exceeds
available open shares.
"""
function close_qty!(ledger::MyTaxLedger, ticker::String, qty_to_close::Int,
        price::Float64, date::Date)
    queue = get(ledger.lots, ticker, MyTaxLot[])
    remaining = qty_to_close
    while remaining > 0 && !isempty(queue)
        front = queue[1]
        take = min(front.qty, remaining)
        holding_days = (date - front.open_date).value
        pnl = take * (price - front.open_price)
        classification = (holding_days >= 365) ? :lt : :st
        if classification == :lt
            ledger.realized_lt_pnl += pnl
        else
            ledger.realized_st_pnl += pnl
        end
        push!(ledger.closed_lots,
              (ticker = ticker, open_date = front.open_date,
               close_date = date, qty = take, pnl = pnl,
               classification = classification,
               holding_days = holding_days))
        if take == front.qty
            popfirst!(queue)
        else
            front.qty -= take
        end
        remaining -= take
    end
    remaining > 0 && error("close_qty!: attempted to close more shares than open for $ticker")
    return nothing
end
```

- [ ] **Step 4: Include and export**

Uncomment `include("Tax.jl")` and add:

```julia
export open_lot!, close_qty!
```

- [ ] **Step 5: Wire into runtests.jl** (`include("test_tax.jl")`)

- [ ] **Step 6: Run tests; expect pass**

- [ ] **Step 7: Commit**

```bash
git add code/src/Tax.jl code/src/ConstrainedCobbDouglas.jl \
        code/test/test_tax.jl code/test/runtests.jl
git commit -m "Tax.jl: lot-by-lot FIFO ledger with ST/LT classification"
```

---

### Task 13: Implement `summarize_after_tax`

**Files:**
- Modify: `code/src/Tax.jl`
- Modify: `code/test/test_tax.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl`

- [ ] **Step 1: Append failing test**

```julia
@testset "summarize_after_tax — symmetric model" begin
    led = MyTaxLedger()
    open_lot!(led, "X", 100, 50.0, Date(2025,1,2))
    close_qty!(led, "X", 100, 60.0, Date(2025,3,1))    # +$1000 ST
    open_lot!(led, "Y", 100, 100.0, Date(2024,1,2))
    close_qty!(led, "Y", 100, 200.0, Date(2025,3,2))   # +$10000 LT (>365 days)
    s = summarize_after_tax(led, (st = 0.37, lt = 0.20))
    @test s.realized_st_pnl ≈ 1000.0
    @test s.realized_lt_pnl ≈ 10_000.0
    @test s.tax_st ≈ 370.0
    @test s.tax_lt ≈ 2_000.0
    @test s.after_tax_realized_pnl ≈ 1000.0 + 10_000.0 - 370.0 - 2000.0
    @test s.lt_share_of_realized ≈ 10_000.0 / 11_000.0
    @test length(s.holding_period_distribution) == 2
end
```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Append to `code/src/Tax.jl`**

```julia
"""
    summarize_after_tax(ledger, rates::NamedTuple) -> NamedTuple

Symmetric tax model: losses generate credits at the category rate.
"""
function summarize_after_tax(ledger::MyTaxLedger,
        rates::NamedTuple)::NamedTuple
    tax_st = rates.st * ledger.realized_st_pnl
    tax_lt = rates.lt * ledger.realized_lt_pnl
    total_tax = tax_st + tax_lt
    realized = ledger.realized_st_pnl + ledger.realized_lt_pnl
    lt_share = realized != 0.0 ? ledger.realized_lt_pnl / realized : 0.0
    hp = [lot.holding_days for lot in ledger.closed_lots]
    return (
        realized_st_pnl = ledger.realized_st_pnl,
        realized_lt_pnl = ledger.realized_lt_pnl,
        tax_st = tax_st,
        tax_lt = tax_lt,
        total_tax = total_tax,
        after_tax_realized_pnl = realized - total_tax,
        lt_share_of_realized = lt_share,
        holding_period_distribution = hp)
end
```

- [ ] **Step 4: Export**

```julia
export summarize_after_tax
```

- [ ] **Step 5: Run tests; expect pass**

- [ ] **Step 6: Commit**

```bash
git add code/src/Tax.jl code/src/ConstrainedCobbDouglas.jl code/test/test_tax.jl
git commit -m "Tax.jl: summarize_after_tax with symmetric loss-credit model"
```

---

## Phase 6 — MPC layer

### Task 14: Implement `forward_project` (Monte Carlo + closed-form arms)

**Files:**
- Create: `code/src/MPC.jl`
- Create: `code/test/test_mpc.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl`
- Modify: `code/test/runtests.jl`

- [ ] **Step 1: Write failing tests**

Create `code/test/test_mpc.jl`:

```julia
using Test
using LinearAlgebra
using Random
using Statistics
using JLD2
using JumpHMM
using ConstrainedCobbDouglas

@testset "MPC module" begin
    # Build a small synthetic market model in-place
    Random.seed!(7)
    market_model = nothing  # set in the test below if available
    # We'll test the asset-projection arm with a fixed market path injection
    # via a helper that's documented inline.

    @testset "Closed-form arm: single-asset moments" begin
        α = 0.05; β = 1.0; σ_ε = 0.20; σ_m = 0.15
        T = 21; Δt = 1.0 / 252.0
        # one-asset projection: V_T closed-form
        w = [1.0]
        μ_arr, σ_arr = forward_project_closed_form(
            [α], [β], σ_m, [σ_ε], w, 100_000.0, T, Δt)
        @test length(μ_arr) == T
        # Variance per Δt = β²σ_m² + σ_ε² ; over T steps: var(log V_T/V_0) = ... · T·Δt
        var_per_step = β^2 * σ_m^2 + σ_ε^2
        var_total = var_per_step * T * Δt
        # σ_T (in dollar units) ≈ V_0 · sqrt(var_total) for small variance
        @test σ_arr[end] > 0.0
    end
end
```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Create `code/src/MPC.jl`**

```julia
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
```

- [ ] **Step 4: Include and export from module**

Uncomment `include("MPC.jl")` and add:

```julia
export forward_project, forward_project_closed_form
```

- [ ] **Step 5: Wire into runtests.jl** (`include("test_mpc.jl")`)

- [ ] **Step 6: Run tests; expect pass**

- [ ] **Step 7: Commit**

```bash
git add code/src/MPC.jl code/src/ConstrainedCobbDouglas.jl \
        code/test/test_mpc.jl code/test/runtests.jl
git commit -m "MPC.jl: forward_project (JumpHMM-SIM hybrid + closed-form arm)"
```

---

### Task 15: Implement `check_trigger` with all three conditions

**Files:**
- Modify: `code/src/MPC.jl`
- Modify: `code/test/test_mpc.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl`

- [ ] **Step 1: Append failing tests**

```julia
@testset "check_trigger: band exit" begin
    proj = MyMPCProjection(
        μ = [100.0, 101.0, 102.0],
        σ = [1.0, 1.0, 1.0],
        V₀ = 100.0, paths = zeros(1, 3),
        decision_date_idx = 1,
        closed_form_μ = [100.0, 101.0, 102.0],
        closed_form_σ = [1.0, 1.0, 1.0],
        divergence_warning = false)
    spec = MyMPCSpec(z = 1.96, T = 21, N = 10, D_max = 0.20)
    state = MyBacktestState()
    state.date_idx = 3; state.last_decision_t = 1
    state.V_t = 110.0    # well above μ + zσ at τ=2
    state.wealth_peak = 110.0
    state.last_projection = proj
    trig = check_trigger(state, spec)
    @test trig.fired == true
    @test trig.reason == :band_exit
end

@testset "check_trigger: horizon elapsed" begin
    proj = MyMPCProjection(
        μ = fill(100.0, 21), σ = fill(1.0, 21),
        V₀ = 100.0, paths = zeros(1, 21),
        decision_date_idx = 1,
        closed_form_μ = fill(100.0, 21),
        closed_form_σ = fill(1.0, 21),
        divergence_warning = false)
    spec = MyMPCSpec(z = 1.96, T = 21, N = 10, D_max = 0.20)
    state = MyBacktestState()
    state.date_idx = 22; state.last_decision_t = 1   # τ = 21
    state.V_t = 100.0; state.wealth_peak = 100.0
    state.last_projection = proj
    trig = check_trigger(state, spec)
    @test trig.fired == true
    @test trig.reason == :horizon_elapsed
end

@testset "check_trigger: drawdown circuit breaker" begin
    proj = MyMPCProjection(
        μ = fill(100.0, 21), σ = fill(50.0, 21),   # very wide band
        V₀ = 100.0, paths = zeros(1, 21),
        decision_date_idx = 1,
        closed_form_μ = fill(100.0, 21),
        closed_form_σ = fill(50.0, 21),
        divergence_warning = false)
    spec = MyMPCSpec(z = 1.96, T = 21, N = 10, D_max = 0.08)
    state = MyBacktestState()
    state.date_idx = 2; state.last_decision_t = 1
    state.V_t = 90.0; state.wealth_peak = 100.0    # 10% drawdown
    state.last_projection = proj
    trig = check_trigger(state, spec)
    @test trig.fired == true
    @test trig.reason == :drawdown
end

@testset "check_trigger: in-spec idle" begin
    proj = MyMPCProjection(
        μ = fill(100.0, 21), σ = fill(2.0, 21),
        V₀ = 100.0, paths = zeros(1, 21),
        decision_date_idx = 1,
        closed_form_μ = fill(100.0, 21),
        closed_form_σ = fill(2.0, 21),
        divergence_warning = false)
    spec = MyMPCSpec(z = 1.96, T = 21, N = 10, D_max = 0.20)
    state = MyBacktestState()
    state.date_idx = 3; state.last_decision_t = 1
    state.V_t = 100.5; state.wealth_peak = 100.5
    state.last_projection = proj
    trig = check_trigger(state, spec)
    @test trig.fired == false
    @test trig.reason == :in_spec
end
```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Append to `code/src/MPC.jl`**

```julia
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
```

- [ ] **Step 4: Export**

```julia
export check_trigger
```

- [ ] **Step 5: Run tests; expect pass**

- [ ] **Step 6: Commit**

```bash
git add code/src/MPC.jl code/src/ConstrainedCobbDouglas.jl code/test/test_mpc.jl
git commit -m "MPC.jl: check_trigger with band-exit / horizon / drawdown conditions"
```

---

## Phase 7 — Bandit module

### Task 16: Per-sector bandit core (sample_arm, quotas, sector-relative reward, training loop)

**Files:**
- Create: `code/src/Bandit.jl`
- Create: `code/test/test_bandit.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl`
- Modify: `code/test/runtests.jl`

- [ ] **Step 1: Write failing tests**

Create `code/test/test_bandit.jl`:

```julia
using Test
using Random
using Statistics
using ConstrainedCobbDouglas

@testset "Bandit module" begin
    @testset "assign_quotas sums to K_basket for uniform 11 sectors" begin
        sector_groups = Dict("S$i" => collect((i-1)*40+1:i*40) for i in 1:11)
        q = assign_quotas(sector_groups, 22)
        @test sum(values(q)) == 22
        @test all(values(q) .== 2)
    end

    @testset "sample_without_replacement picks distinct elements" begin
        rng = MersenneTwister(1)
        pool = collect(1:50)
        s = ConstrainedCobbDouglas.sample_without_replacement(rng, pool, 5)
        @test length(s) == 5
        @test length(unique(s)) == 5
        @test all(x in pool for x in s)
    end

    @testset "ε decay monotone-decreasing until floor" begin
        n_arms = 1000
        ε_floor = 0.05
        ε(t) = max(ε_floor, t > 1 ? min(1.0, t^(-1/3) * (n_arms * log(t))^(1/3)) : 1.0)
        seq = [ε(t) for t in 2:5000]
        @test seq[end] == ε_floor
        @test issorted(seq; rev = true) || all(diff(seq) .<= 1e-12)
    end
end
```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Create `code/src/Bandit.jl`**

```julia
"""
    assign_quotas(sector_groups, K_total) -> Dict{String,Int}

Equal-weight quotas with bonus to largest sectors so Σ q_s = K_total.
"""
function assign_quotas(sector_groups::Dict{String,Vector{Int}},
        K_total::Int)::Dict{String,Int}
    sectors = collect(keys(sector_groups))
    S = length(sectors)
    base = K_total ÷ S
    remainder = K_total - base * S
    sorted = sort(sectors; by = s -> -length(sector_groups[s]))
    q = Dict{String,Int}()
    for (rank, s) in enumerate(sorted)
        q[s] = base + (rank <= remainder ? 1 : 0)
    end
    return q
end

"""
    sample_without_replacement(rng, pool, k) -> Vector{Int}
"""
function sample_without_replacement(rng::AbstractRNG, pool::Vector{Int},
        k::Int)::Vector{Int}
    k >= length(pool) && return copy(pool)
    return shuffle(rng, pool)[1:k]
end

"""
    cd_basket_return(arm_idx, day, horizon, price_matrix, sim_params, tickers) -> Float64

Cobb-Douglas-allocated buy-and-hold log return of `arm_idx` (column indices
into `price_matrix`) over `[day, day+horizon]`. Uses analytical CD allocator
with γ = compute_preference_weights at `day`.
"""
function cd_basket_return(arm_idx::Vector{Int}, day::Int, horizon::Int,
        price_matrix::Matrix{Float64}, sim_params::Dict{String,Tuple{Float64,Float64,Float64}},
        tickers::Vector{String}, gm_t::Float64, lambda::Float64;
        B::Float64 = 100_000.0)::Float64
    arm_tickers = tickers[arm_idx]
    γ = compute_preference_weights(sim_params, arm_tickers, gm_t, lambda)
    p_d = price_matrix[day, arm_idx]
    any(p_d .<= 0.0) && return 0.0
    n, cash = solve_unconstrained_cd_analytical(Vector{Float64}(γ), Vector{Float64}(p_d), B)
    p_dh = price_matrix[day + horizon, arm_idx]
    any(p_dh .<= 0.0) && return 0.0
    W_dh = sum(n .* p_dh) + cash
    return log(W_dh / B)
end

"""
    sector_ew_log_return(sector_idx, day, horizon, price_matrix) -> Float64
"""
function sector_ew_log_return(sector_idx::Vector{Int}, day::Int, horizon::Int,
        price_matrix::Matrix{Float64})::Float64
    p_d  = price_matrix[day, sector_idx]
    p_dh = price_matrix[day + horizon, sector_idx]
    (any(p_d .<= 0.0) || any(p_dh .<= 0.0)) && return 0.0
    return log(mean(p_dh ./ p_d))
end

"""
    sector_relative_reward(arm, sector_idx, day, horizon, price_matrix, sim_params,
                           tickers, gm_t, lambda) -> Float64

Cross-sectional alpha: CD-allocated basket log return minus sector EW log return.
"""
function sector_relative_reward(arm::Vector{Int}, sector_idx::Vector{Int},
        day::Int, horizon::Int, price_matrix::Matrix{Float64},
        sim_params::Dict{String,Tuple{Float64,Float64,Float64}},
        tickers::Vector{String}, gm_t::Float64, lambda::Float64)::Float64
    r_basket = cd_basket_return(arm, day, horizon, price_matrix, sim_params,
                                tickers, gm_t, lambda)
    r_sector = sector_ew_log_return(sector_idx, day, horizon, price_matrix)
    return r_basket - r_sector
end

"""
    train_sector_bandit(sector_idx, q, train_offset, train_last, horizon,
                        price_matrix, sim_params, tickers, gm, λ_series;
                        iters, seed, ε_floor) -> NamedTuple

ε-greedy bandit on a single sector. Returns (best_arm, best_mean, rewards,
n_arms, n_unique).
"""
function train_sector_bandit(sector_idx::Vector{Int}, q::Int,
        train_offset::Int, train_last::Int, horizon::Int,
        price_matrix::Matrix{Float64},
        sim_params::Dict{String,Tuple{Float64,Float64,Float64}},
        tickers::Vector{String}, gm::Vector{Float64}, λ_series::Vector{Float64};
        iters::Int, seed::Int, ε_floor::Float64 = 0.05)::NamedTuple
    rng = MersenneTwister(seed)
    N_s = length(sector_idx)
    n_arms = binomial(N_s, q)
    arm_mean = Dict{Vector{Int},Float64}()
    arm_count = Dict{Vector{Int},Int}()
    rewards = zeros(Float64, iters)
    for t in 1:iters
        ε = max(ε_floor,
            t > 1 ? min(1.0, t^(-1/3) * (n_arms * log(t))^(1/3)) : 1.0)
        arm = if rand(rng) < ε || isempty(arm_mean)
            sort(sample_without_replacement(rng, sector_idx, q))
        else
            argmax(arm_mean)::Vector{Int}
        end
        day = rand(rng, train_offset:train_last)
        gm_t = gm[day]; lambda = λ_series[day]
        r = sector_relative_reward(arm, sector_idx, day, horizon,
            price_matrix, sim_params, tickers, gm_t, lambda)
        c = get(arm_count, arm, 0) + 1
        m = get(arm_mean, arm, 0.0)
        arm_mean[arm] = m + (r - m) / c
        arm_count[arm] = c
        rewards[t] = r
    end
    best_arm = argmax(arm_mean)
    return (best_arm = best_arm,
            best_mean = arm_mean[best_arm],
            rewards = rewards,
            n_arms = n_arms,
            n_unique = length(arm_mean))
end
```

- [ ] **Step 4: Include and export from module**

Uncomment `include("Bandit.jl")` and add:

```julia
export assign_quotas, sector_relative_reward, train_sector_bandit, cd_basket_return
```

- [ ] **Step 5: Wire into runtests.jl** (`include("test_bandit.jl")`)

- [ ] **Step 6: Run tests; expect pass**

- [ ] **Step 7: Commit**

```bash
git add code/src/Bandit.jl code/src/ConstrainedCobbDouglas.jl \
        code/test/test_bandit.jl code/test/runtests.jl
git commit -m "Bandit.jl: per-sector ε-greedy core (quotas, sector-relative reward, training)"
```

---

## Phase 8 — Files module + materialize_orders

### Task 17: Implement `Files.jl` I/O helpers

**Files:**
- Create: `code/src/Files.jl`
- Create: `code/test/test_files.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl`
- Modify: `code/test/runtests.jl`

- [ ] **Step 1: Write failing tests**

Create `code/test/test_files.jl`:

```julia
using Test
using JLD2
using CSV
using DataFrames
using ConstrainedCobbDouglas

@testset "Files module" begin
    @testset "load_sector_map returns Dict and detects mismatches" begin
        csv_path = joinpath(@__DIR__, "..", "src", "data", "sp500-sectors.csv")
        sector_of, dropped = load_sector_map(["AAPL", "MSFT", "ZZZZZ_NOT_REAL"], csv_path)
        @test haskey(sector_of, "AAPL")
        @test haskey(sector_of, "MSFT")
        @test "ZZZZZ_NOT_REAL" in dropped
    end

    @testset "save_results / load_results round-trip" begin
        tmp = tempname() * ".jld2"
        d = Dict("a" => [1.0, 2.0], "b" => "hello", "c" => 42)
        save_results(tmp, d)
        d2 = load_results(tmp)
        @test d2["a"] == d["a"]
        @test d2["b"] == d["b"]
        @test d2["c"] == d["c"]
        rm(tmp)
    end
end
```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Create `code/src/Files.jl`**

```julia
"""
    load_ohlc_jld2(path) -> NamedTuple

Load a lectures-style OHLC JLD2 file. Returns (prices::Matrix, dates::Vector{Date},
tickers::Vector{String}, volumes::Matrix). Schema follows what the lectures
repo writes; missing fields are returned as empty.
"""
function load_ohlc_jld2(path::String)::NamedTuple
    d = load(path)
    # Resolve the actual stored field names (lectures uses several layouts).
    function pick(keys...)
        for k in keys
            haskey(d, k) && return d[k]
        end
        return nothing
    end
    prices  = pick("close", "prices", "Close")
    dates   = pick("dates", "Date")
    tickers = pick("tickers", "symbols", "Symbol")
    volumes = pick("volume", "Volume")
    return (prices = prices, dates = dates, tickers = tickers, volumes = volumes)
end

"""
    load_sector_map(tickers, csv_path) -> (Dict{String,String}, Vector{String})

Read S&P 500 sector CSV and produce a ticker→sector map plus a list of
unmatched tickers.
"""
function load_sector_map(tickers::Vector{String},
        csv_path::String)::Tuple{Dict{String,String},Vector{String}}
    df = CSV.read(csv_path, DataFrame)
    sym_col = :Symbol
    sec_col = Symbol("GICS Sector")
    lookup = Dict{String,String}(
        row[sym_col] => row[sec_col] for row in eachrow(df))
    sector_of = Dict{String,String}()
    dropped = String[]
    for t in tickers
        if haskey(lookup, t)
            sector_of[t] = lookup[t]
        else
            push!(dropped, t)
        end
    end
    return sector_of, dropped
end

"""
    save_results(path, dict::Dict{String,Any})
"""
save_results(path::String, d::Dict) = jldsave(path; d...)

"""
    load_results(path) -> Dict
"""
load_results(path::String) = load(path)
```

- [ ] **Step 4: Include and export**

Uncomment `include("Files.jl")` and add:

```julia
export load_ohlc_jld2, load_sector_map, save_results, load_results
```

- [ ] **Step 5: Wire into runtests.jl** (`include("test_files.jl")`)

- [ ] **Step 6: Run tests; expect pass**

- [ ] **Step 7: Commit**

```bash
git add code/src/Files.jl code/src/ConstrainedCobbDouglas.jl \
        code/test/test_files.jl code/test/runtests.jl
git commit -m "Files.jl: OHLC + sector + JLD2 I/O helpers"
```

---

### Task 18: Implement `materialize_orders`

**Files:**
- Create: `code/src/Backtest.jl`
- Create: `code/test/test_backtest.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl`
- Modify: `code/test/runtests.jl`

- [ ] **Step 1: Write failing tests**

Create `code/test/test_backtest.jl`:

```julia
using Test
using Dates
using ConstrainedCobbDouglas

@testset "Backtest module" begin
    @testset "materialize_orders rounds and respects min-order" begin
        cm = MyCostModel(commission_per_trade = 0.0, half_spread_bps = 5.0,
                         slippage_κ = 0.001, adv = Dict("A"=>1e9, "B"=>1e9))
        n_target = [99.6, 50.4]
        n_current = [100.0, 50.0]
        prices = [100.0, 50.0]
        orders = materialize_orders(["A","B"], n_target, n_current, prices,
                                    1e6, cm; min_dollar = 1000.0)
        # qty deltas: -0.4, +0.4 → after round: 0, 0 → all suppressed by min_dollar
        @test isempty(orders)
    end

    @testset "materialize_orders generates an order above threshold" begin
        cm = MyCostModel(commission_per_trade = 0.0, half_spread_bps = 5.0,
                         slippage_κ = 0.001, adv = Dict("A"=>1e9))
        orders = materialize_orders(["A"], [120.0], [100.0], [100.0],
                                    1e6, cm; min_dollar = 100.0)
        @test length(orders) == 1
        @test orders[1].ticker == "A"
        @test orders[1].qty == 20
    end
end
```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Create `code/src/Backtest.jl`** with the helper

```julia
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
```

- [ ] **Step 4: Include and export from module**

Uncomment `include("Backtest.jl")` and add:

```julia
export materialize_orders
```

- [ ] **Step 5: Wire into runtests.jl** (`include("test_backtest.jl")`)

- [ ] **Step 6: Run tests; expect pass**

- [ ] **Step 7: Commit**

```bash
git add code/src/Backtest.jl code/src/ConstrainedCobbDouglas.jl \
        code/test/test_backtest.jl code/test/runtests.jl
git commit -m "Backtest.jl: materialize_orders with min-dollar suppression of jitter"
```

---

## Phase 9 — Backtest harness

### Task 19: Implement `should_decide` and `allocate` dispatch for all 6 strategies

**Files:**
- Modify: `code/src/Backtest.jl`
- Modify: `code/test/test_backtest.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl`

- [ ] **Step 1: Append failing tests**

```julia
@testset "should_decide: buy-and-hold strategies only fire at t=1" begin
    state = MyBacktestState()
    state.date_idx = 1
    @test should_decide(EqualWeightStrategy(), state, 1) == true
    @test should_decide(MinVarBuyHoldStrategy(), state, 1) == true
    state.date_idx = 5
    @test should_decide(EqualWeightStrategy(), state, 5) == false
    @test should_decide(MinVarBuyHoldStrategy(), state, 5) == false
end

@testset "should_decide: daily strategies fire every day" begin
    state = MyBacktestState()
    state.date_idx = 50
    @test should_decide(UnconstrainedCDStrategy(), state, 50) == true
    @test should_decide(CostAwareMVStrategy(κ = 5.0, c = 0.001), state, 50) == true
end

@testset "should_decide: MPC strategies fire on trigger or day 1" begin
    spec = MyMPCSpec(z = 1.96, T = 21, N = 100, D_max = 0.20)
    s5 = CDWithMPCStrategy(spec = spec)
    state = MyBacktestState()
    state.date_idx = 1; state.next_decision_due = false
    @test should_decide(s5, state, 1) == true   # initial allocation
    state.date_idx = 5; state.next_decision_due = false
    @test should_decide(s5, state, 5) == false
    state.next_decision_due = true
    @test should_decide(s5, state, 5) == true
end
```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Append to `code/src/Backtest.jl`**

```julia
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
```

- [ ] **Step 4: Export**

```julia
export should_decide, allocate
```

- [ ] **Step 5: Run tests; expect pass**

- [ ] **Step 6: Commit**

```bash
git add code/src/Backtest.jl code/src/ConstrainedCobbDouglas.jl code/test/test_backtest.jl
git commit -m "Backtest.jl: should_decide + allocate dispatch for all 6 strategies"
```

---

### Task 20: Implement `run_backtest` core loop with EWLS update + cost/tax integration

**Files:**
- Modify: `code/src/Backtest.jl`
- Modify: `code/test/test_backtest.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl`

- [ ] **Step 1: Append a failing test for strategy isolation**

```julia
@testset "run_backtest: EqualWeightStrategy on synthetic prices is reproducible" begin
    using Random
    Random.seed!(1)
    K = 4; n_days = 30
    tickers = ["A","B","C","D"]
    prices = zeros(n_days, K)
    prices[1, :] = [100.0, 50.0, 25.0, 200.0]
    for t in 2:n_days, i in 1:K
        prices[t, i] = prices[t-1, i] * (1.0 + 0.0005 + 0.001 * randn())
    end
    volumes = fill(1.0e9, n_days, K)
    market_prices = vec(mean(prices; dims = 2))
    α = fill(0.0, K); β = fill(1.0, K); σ_ε = fill(0.1, K); σ_m = 0.10
    sim_params = Dict(tickers[i] => (α[i], β[i], σ_ε[i]) for i in 1:K)
    sim_init = Dict(tickers[i] =>
        ewls_init(α[i], β[i], σ_ε[i]; half_life = 21.0, prior_weight = 21.0)
        for i in 1:K)
    env = (tickers = tickers, prices = prices, market_prices = market_prices,
           volumes = volumes, sim_params_init = sim_init,
           σ_m = σ_m, dates = [Date(2025,1,2) + Day(t-1) for t in 1:n_days],
           market_model = nothing, c̄ = 0.05)
    cm = MyCostModel(commission_per_trade = 0.0, half_spread_bps = 5.0,
                     slippage_κ = 0.001,
                     adv = Dict(t => 1e9 for t in tickers))
    rates = (st = 0.37, lt = 0.20)
    res1 = run_backtest(EqualWeightStrategy(), env, cm, rates;
                        B₀ = 100_000.0, rng_seed = 42)
    res2 = run_backtest(EqualWeightStrategy(), env, cm, rates;
                        B₀ = 100_000.0, rng_seed = 42)
    @test res1.wealth_after_cost_pretax == res2.wealth_after_cost_pretax
    @test length(res1.wealth_after_cost_pretax) == n_days
end
```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Append to `code/src/Backtest.jl`**

```julia
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
    # Shadow gross wealth tracker (no costs deducted)
    gross_cash = B₀

    is_mpc(s::MyAllocationStrategy) = s isa CDWithMPCStrategy || s isa ConstrainedCDWithMPCStrategy

    for t in 1:n_days
        state.date_idx = t
        state.prices = env.prices[t, :]
        # Mark-to-market
        state.V_t = sum(state.positions .* state.prices) + state.cash
        gross_V = sum(state.positions .* state.prices) + gross_cash
        state.wealth_peak = max(state.wealth_peak, state.V_t)
        state.just_decided = false

        # Build env_for_step (current γ, Σ from EWLS state)
        αs = [state.sim_state[tk].α for tk in tickers]
        βs = [state.sim_state[tk].β for tk in tickers]
        σ_εs = [state.sim_state[tk].σ_ε for tk in tickers]
        # γ uses no-news formula via Dict
        sim_params_now = Dict(tickers[i] => (αs[i], βs[i], σ_εs[i]) for i in 1:K)
        # Compute gm_t and λ_t from the rolling market window
        win = max(1, t - 63)
        mkt_window = env.market_prices[win:t]
        if length(mkt_window) >= 2
            mkt_growth = compute_market_growth(Vector{Float64}(mkt_window))
            gm_t = isempty(mkt_growth) ? 0.0 : mkt_growth[end]
        else
            gm_t = 0.0
        end
        # λ via short/long EMA on prices window
        ema_window = length(mkt_window) >= 63 ? mkt_window : Vector{Float64}(mkt_window)
        if length(ema_window) >= 21
            short = compute_ema(Vector{Float64}(ema_window); window = 21)
            long  = compute_ema(Vector{Float64}(ema_window); window = max(21, length(ema_window)))
            λ_t = compute_lambda(short, long)[end]
        else
            λ_t = 0.5
        end
        γ_t = compute_preference_weights(sim_params_now, tickers, gm_t, λ_t)
        # SIM-implied Σ at this state
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
                gross_cash -= 0.0   # gross tracker doesn't pay cost
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
                # Build env_for_proj with α/β/σ_ε aligned to position vector
                env_proj = merge(env_step, (market_model = env.market_model,))
                state.last_projection = forward_project(state,
                    (strategy isa CDWithMPCStrategy ? strategy.spec : strategy.spec),
                    env_proj)
                state.last_decision_t = t
            end
        end

        # MPC trigger check on non-decision days
        if is_mpc(strategy) && !state.just_decided && state.last_projection !== nothing
            spec = (strategy isa CDWithMPCStrategy ? strategy.spec : strategy.spec)
            trig = check_trigger(state, spec)
            push!(state.trigger_log, trig)
            if trig.fired
                state.next_decision_due = true
            end
        end

        # EWLS update — every day, regardless of decision
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

        # Record history
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
    # After-tax wealth: deduct cumulative tax (one-shot at terminal, mirrored back)
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
```

- [ ] **Step 4: Export**

```julia
export run_backtest, summary_metrics
```

- [ ] **Step 5: Run tests; expect pass**

- [ ] **Step 6: Commit**

```bash
git add code/src/Backtest.jl code/src/ConstrainedCobbDouglas.jl code/test/test_backtest.jl
git commit -m "Backtest.jl: run_backtest core loop with EWLS rolling SIM"
```

---

### Task 21: Implement `compare_strategies` orchestrator

**Files:**
- Modify: `code/src/Backtest.jl`
- Modify: `code/test/test_backtest.jl`
- Modify: `code/src/ConstrainedCobbDouglas.jl`

- [ ] **Step 1: Append failing test**

```julia
@testset "compare_strategies runs all strategies sequentially" begin
    using Random
    Random.seed!(2)
    K = 4; n_days = 30
    tickers = ["A","B","C","D"]
    prices = zeros(n_days, K)
    prices[1, :] = [100.0, 50.0, 25.0, 200.0]
    for t in 2:n_days, i in 1:K
        prices[t, i] = prices[t-1, i] * (1.0 + 0.0005 + 0.001 * randn())
    end
    sim_init = Dict(tickers[i] => ewls_init(0.0, 1.0, 0.1; half_life = 21.0, prior_weight = 21.0)
                    for i in 1:K)
    env = (tickers = tickers, prices = prices,
           market_prices = vec(mean(prices; dims = 2)),
           volumes = fill(1e9, n_days, K),
           sim_params_init = sim_init, σ_m = 0.10,
           dates = [Date(2025,1,2) + Day(t-1) for t in 1:n_days],
           market_model = nothing, c̄ = 0.05)
    cm = MyCostModel(commission_per_trade = 0.0, half_spread_bps = 5.0,
                     slippage_κ = 0.001,
                     adv = Dict(t => 1e9 for t in tickers))
    rates = (st = 0.37, lt = 0.20)
    strategies = MyAllocationStrategy[
        EqualWeightStrategy(),
        UnconstrainedCDStrategy()]
    results = compare_strategies(strategies, env, cm, rates; B₀ = 100_000.0, rng_seed = 42)
    @test haskey(results, "EqualWeightStrategy")
    @test haskey(results, "UnconstrainedCDStrategy")
    @test results["EqualWeightStrategy"].wealth_after_cost_pretax[1] == 100_000.0
end
```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Append to `code/src/Backtest.jl`**

```julia
"""
    compare_strategies(strategies, env, cost_model, tax_rates;
                       B₀ = 100_000.0, rng_seed = 42, parallel = false)
                       -> Dict{String, MyBacktestResult}
"""
function compare_strategies(strategies::Vector{<:MyAllocationStrategy}, env,
        cost_model::MyCostModel, tax_rates::NamedTuple;
        B₀::Float64 = 100_000.0, rng_seed::Int = 42,
        parallel::Bool = false)::Dict{String,MyBacktestResult}
    results = Dict{String,MyBacktestResult}()
    for (i, s) in enumerate(strategies)
        name = string(typeof(s).name.name)
        seed_i = rng_seed + 1000 * i
        results[name] = run_backtest(s, env, cost_model, tax_rates;
                                     B₀ = B₀, rng_seed = seed_i)
    end
    return results
end
```

- [ ] **Step 4: Export**

```julia
export compare_strategies
```

- [ ] **Step 5: Run tests; expect pass**

- [ ] **Step 6: Commit**

```bash
git add code/src/Backtest.jl code/src/ConstrainedCobbDouglas.jl code/test/test_backtest.jl
git commit -m "Backtest.jl: compare_strategies orchestrator"
```

---

## Phase 10 — Scripts

### Task 22: Script 01 — calibrate SIM on 2014-2024

**Files:**
- Create: `scripts/01_calibrate_sim.jl`

- [ ] **Step 1: Create `scripts/01_calibrate_sim.jl`**

```julia
# scripts/01_calibrate_sim.jl
# Fit per-ticker SIM on 2014-2024 daily closes; compute ADV; write
# scripts/data/sim_calibration.jld2.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "code"))

using ConstrainedCobbDouglas
using JLD2
using Statistics

const PATH_INPUTS = joinpath(@__DIR__, "..", "code", "src", "data")
const PATH_OUT = joinpath(@__DIR__, "data")
isdir(PATH_OUT) || mkpath(PATH_OUT)

const SIM_SEED = 2026

println("=" ^ 78)
println("01_calibrate_sim.jl — fitting SIM on 2014-2024")
println("=" ^ 78)
versioninfo()

# Load training-window OHLC
ohlc_train = load_ohlc_jld2(joinpath(PATH_INPUTS,
    "SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2"))
prices_train = ohlc_train.prices            # n_days × K
tickers_all  = ohlc_train.tickers
volumes_train = ohlc_train.volumes
n_days, K_all = size(prices_train)
println("Loaded $n_days × $K_all training matrix")

# Load hold-out OHLC to enforce the universe filter
ohlc_h1 = load_ohlc_jld2(joinpath(PATH_INPUTS,
    "SP500-Daily-OHLC-1-2-2025-to-12-31-2025.jld2"))
ohlc_h2 = load_ohlc_jld2(joinpath(PATH_INPUTS,
    "SP500-Daily-OHLC-1-2-2026-to-04-22-2026.jld2"))

function full_coverage(prices)
    ok = trues(size(prices, 2))
    for i in axes(prices, 2)
        col = prices[:, i]
        if any(ismissing, col) || any(c -> !isfinite(c) || c <= 0.0, col)
            ok[i] = false
        end
    end
    return ok
end
ok_train = full_coverage(prices_train)
ok_h1    = full_coverage(ohlc_h1.prices)
ok_h2    = full_coverage(ohlc_h2.prices)
keep = ok_train .& ok_h1 .& ok_h2
tickers = tickers_all[keep]
prices  = prices_train[:, keep]
volumes = volumes_train[:, keep]
println("Universe filter: $(length(tickers)) of $K_all tickers survive full coverage")

# Pick market index: first column with ticker "SPY", else equal-weight synthetic
market_idx = findfirst(==("SPY"), tickers)
market_prices = market_idx === nothing ?
    vec(mean(prices; dims = 2)) :
    prices[:, market_idx]

g_m = compute_market_growth(Vector{Float64}(market_prices))
σ_m = std(g_m)
println("σ_m (annualized) = ", round(σ_m; digits = 4))

K = length(tickers)
αs = zeros(K); βs = zeros(K); σ_εs = zeros(K); r²s = zeros(K)
for (i, tk) in enumerate(tickers)
    g_i = compute_market_growth(Vector{Float64}(prices[:, i]))
    n_use = min(length(g_m), length(g_i))
    est = estimate_sim(g_m[1:n_use], g_i[1:n_use], tk)
    αs[i] = est.α; βs[i] = est.β; σ_εs[i] = est.σ_ε; r²s[i] = est.r²
end

adv = Dict(tickers[i] => mean(skipmissing(volumes[:, i])) for i in 1:K)

out = Dict(
    "config" => Dict("SIM_SEED" => SIM_SEED),
    "tickers" => collect(tickers),
    "alpha" => αs, "beta" => βs, "sigma_eps" => σ_εs, "r_squared" => r²s,
    "sigma_market" => σ_m,
    "adv" => adv,
    "n_training_days" => n_days)

outpath = joinpath(PATH_OUT, "sim_calibration.jld2")
save_results(outpath, out)
println("Saved $outpath")
```

- [ ] **Step 2: Run the script**

```bash
julia --project=code scripts/01_calibrate_sim.jl
```

Expected: prints config + per-ticker count; writes `scripts/data/sim_calibration.jld2`.

- [ ] **Step 3: Sanity-check the artifact**

```bash
julia --project=code -e '
using JLD2
d = load("scripts/data/sim_calibration.jld2")
println("keys: ", keys(d))
println("n tickers: ", length(d["tickers"]))
println("σ_m: ", d["sigma_market"])
'
```

Expected: ~400-413 tickers, σ_m around 0.15-0.25.

- [ ] **Step 4: Commit**

```bash
git add scripts/01_calibrate_sim.jl
git commit -m "scripts/01: calibrate SIM on 2014-2024, write sim_calibration.jld2"
```

---

### Task 23: Scripts 02 + 03 + 04 — bandit single-seed, Monte Carlo, basket selection

**Files:**
- Create: `scripts/02_train_bandit.jl`
- Create: `scripts/03_train_bandit_mc.jl`
- Create: `scripts/04_select_basket.jl`

- [ ] **Step 1: Create `scripts/02_train_bandit.jl`**

```julia
# scripts/02_train_bandit.jl
# Single-seed run of the per-sector bandit. Dev sanity check before MC.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "code"))
using ConstrainedCobbDouglas
using JLD2
using Random
using Statistics

const PATH_INPUTS = joinpath(@__DIR__, "..", "code", "src", "data")
const PATH_OUT = joinpath(@__DIR__, "data")

const BANDIT_SEED = 2026
const K_BASKET = 22
const ITERS_PER_ARM = 50
const ITERS_MAX = 5000
const ITERS_MIN = 500
const FORWARD_HORIZON = 21
const TRAIN_OFFSET = 252         # warm-up before any forward window
println("=" ^ 78)
println("02_train_bandit.jl — single seed = $BANDIT_SEED")
println("=" ^ 78)

# Load SIM calibration + training OHLC
sim_calib = load_results(joinpath(PATH_OUT, "sim_calibration.jld2"))
ohlc_train = load_ohlc_jld2(joinpath(PATH_INPUTS,
    "SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2"))
tickers = sim_calib["tickers"]
prices_full = ohlc_train.prices
tickers_full = ohlc_train.tickers
# Build column index map matching sim_calib tickers
col_of = Dict(t => i for (i, t) in enumerate(tickers_full))
keep_cols = [col_of[t] for t in tickers]
price_matrix = Matrix{Float64}(prices_full[:, keep_cols])

market_idx = findfirst(==("SPY"), tickers)
market_prices = market_idx === nothing ? vec(mean(price_matrix; dims = 2)) : price_matrix[:, market_idx]

g_m = compute_market_growth(Vector{Float64}(market_prices))
short_ema = compute_ema(Vector{Float64}(market_prices); window = 21)
long_ema  = compute_ema(Vector{Float64}(market_prices); window = 63)
λ_series  = compute_lambda(short_ema, long_ema)
# Pad gm to length n_days (compute_market_growth returns n_days - 1)
n_days = length(market_prices)
gm_series = vcat([0.0], g_m)

# Sector map
sector_csv = joinpath(PATH_INPUTS, "sp500-sectors.csv")
sector_of, dropped = load_sector_map(tickers, sector_csv)
sector_groups = Dict{String,Vector{Int}}()
for (i, t) in enumerate(tickers)
    if haskey(sector_of, t)
        push!(get!(sector_groups, sector_of[t], Int[]), i)
    end
end
println("Sectors: $(length(sector_groups));  dropped: $(length(dropped))")

quotas = assign_quotas(sector_groups, K_BASKET)
println("Quotas:")
for s in keys(sector_groups)
    println("  ", rpad(s, 25), "  N_s = ", lpad(length(sector_groups[s]), 3),
        "   q_s = ", quotas[s])
end

# SIM params dict for sector_relative_reward
sim_params = Dict(tickers[i] => (Float64(sim_calib["alpha"][i]),
                                 Float64(sim_calib["beta"][i]),
                                 Float64(sim_calib["sigma_eps"][i]))
                  for i in eachindex(tickers))

train_offset = TRAIN_OFFSET
train_last   = n_days - FORWARD_HORIZON - 1

rng_master = MersenneTwister(BANDIT_SEED)
sector_results = Dict{String,NamedTuple}()
for s in sort(collect(keys(sector_groups)))
    sec_idx = sector_groups[s]
    q = quotas[s]
    n_arms = binomial(length(sec_idx), q)
    iters = clamp(n_arms * ITERS_PER_ARM, ITERS_MIN, ITERS_MAX)
    seed = rand(rng_master, 1:10^9)
    t0 = time()
    res = train_sector_bandit(sec_idx, q, train_offset, train_last,
            FORWARD_HORIZON, price_matrix, sim_params,
            Vector{String}(tickers), gm_series, λ_series;
            iters = iters, seed = seed)
    sector_results[s] = res
    println("  ", rpad(s, 25),
        "  iters=", lpad(iters, 5),
        "  best_mean=", round(res.best_mean; digits = 4),
        "  ", round(time() - t0; digits = 1), "s")
end

# Assemble basket
basket_indices = Int[]
for s in sort(collect(keys(sector_groups)))
    append!(basket_indices, sector_results[s].best_arm)
end
basket_tickers = tickers[basket_indices]
println("\nAssembled basket ($(length(basket_tickers)) tickers): ",
        join(basket_tickers, ", "))

out = Dict(
    "config" => Dict("BANDIT_SEED" => BANDIT_SEED, "K_BASKET" => K_BASKET,
                     "FORWARD_HORIZON" => FORWARD_HORIZON),
    "quotas" => quotas,
    "sector_best_arms" => Dict(s => sector_results[s].best_arm for s in keys(sector_results)),
    "sector_best_means" => Dict(s => sector_results[s].best_mean for s in keys(sector_results)),
    "basket_tickers" => collect(basket_tickers),
    "basket_indices" => collect(basket_indices))
save_results(joinpath(PATH_OUT, "per_sector_bandit_results.jld2"), out)
println("Saved scripts/data/per_sector_bandit_results.jld2")
```

- [ ] **Step 2: Run script 02**

```bash
julia --project=code scripts/02_train_bandit.jl
```

Expected: ~1 min; prints quotas + sector iterations; writes `per_sector_bandit_results.jld2`.

- [ ] **Step 3: Create `scripts/03_train_bandit_mc.jl`** (30 seeds)

```julia
# scripts/03_train_bandit_mc.jl
# 30-seed Monte Carlo. Reuses the per-sector bandit machinery from 02.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "code"))
using ConstrainedCobbDouglas
using JLD2
using Random
using Statistics

const PATH_INPUTS = joinpath(@__DIR__, "..", "code", "src", "data")
const PATH_OUT = joinpath(@__DIR__, "data")
const BANDIT_MC_SEEDS = 1001:1030
const K_BASKET = 22
const ITERS_PER_ARM = 50
const ITERS_MAX = 5000
const ITERS_MIN = 500
const FORWARD_HORIZON = 21
const TRAIN_OFFSET = 252

println("=" ^ 78)
println("03_train_bandit_mc.jl — 30 seeds")
println("=" ^ 78)

sim_calib = load_results(joinpath(PATH_OUT, "sim_calibration.jld2"))
ohlc_train = load_ohlc_jld2(joinpath(PATH_INPUTS,
    "SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2"))
tickers = sim_calib["tickers"]
tickers_full = ohlc_train.tickers
col_of = Dict(t => i for (i, t) in enumerate(tickers_full))
keep_cols = [col_of[t] for t in tickers]
price_matrix = Matrix{Float64}(ohlc_train.prices[:, keep_cols])
market_idx = findfirst(==("SPY"), tickers)
market_prices = market_idx === nothing ? vec(mean(price_matrix; dims = 2)) : price_matrix[:, market_idx]
g_m = compute_market_growth(Vector{Float64}(market_prices))
short_ema = compute_ema(Vector{Float64}(market_prices); window = 21)
long_ema  = compute_ema(Vector{Float64}(market_prices); window = 63)
λ_series  = compute_lambda(short_ema, long_ema)
gm_series = vcat([0.0], g_m)
n_days = length(market_prices)
sector_of, _ = load_sector_map(tickers, joinpath(PATH_INPUTS, "sp500-sectors.csv"))
sector_groups = Dict{String,Vector{Int}}()
for (i, t) in enumerate(tickers)
    haskey(sector_of, t) && push!(get!(sector_groups, sector_of[t], Int[]), i)
end
quotas = assign_quotas(sector_groups, K_BASKET)
sim_params = Dict(tickers[i] => (Float64(sim_calib["alpha"][i]),
                                 Float64(sim_calib["beta"][i]),
                                 Float64(sim_calib["sigma_eps"][i]))
                  for i in eachindex(tickers))
train_offset = TRAIN_OFFSET
train_last = n_days - FORWARD_HORIZON - 1

# Loop seeds
per_seed_tickers = Vector{Vector{String}}()
per_seed_indices = Vector{Vector{Int}}()
per_seed_best_means = Vector{Dict{String,Float64}}()
for (k, seed) in enumerate(BANDIT_MC_SEEDS)
    println("Seed $seed ($(k)/$(length(BANDIT_MC_SEEDS)))")
    rng_master = MersenneTwister(seed)
    sector_results = Dict{String,NamedTuple}()
    for s in sort(collect(keys(sector_groups)))
        sec_idx = sector_groups[s]
        q = quotas[s]
        n_arms = binomial(length(sec_idx), q)
        iters = clamp(n_arms * ITERS_PER_ARM, ITERS_MIN, ITERS_MAX)
        sub_seed = rand(rng_master, 1:10^9)
        res = train_sector_bandit(sec_idx, q, train_offset, train_last,
                FORWARD_HORIZON, price_matrix, sim_params,
                Vector{String}(tickers), gm_series, λ_series;
                iters = iters, seed = sub_seed)
        sector_results[s] = res
    end
    indices = Int[]
    for s in sort(collect(keys(sector_groups)))
        append!(indices, sector_results[s].best_arm)
    end
    push!(per_seed_indices, indices)
    push!(per_seed_tickers, tickers[indices])
    push!(per_seed_best_means,
          Dict(s => sector_results[s].best_mean for s in keys(sector_results)))
end

save_results(joinpath(PATH_OUT, "per_sector_bandit_mc_results.jld2"), Dict(
    "config" => Dict("BANDIT_MC_SEEDS" => collect(BANDIT_MC_SEEDS),
                     "K_BASKET" => K_BASKET, "FORWARD_HORIZON" => FORWARD_HORIZON),
    "quotas" => quotas,
    "per_seed_tickers" => per_seed_tickers,
    "per_seed_indices" => per_seed_indices,
    "per_seed_best_means" => per_seed_best_means))
println("Saved scripts/data/per_sector_bandit_mc_results.jld2")
```

- [ ] **Step 4: Run script 03**

```bash
julia --project=code scripts/03_train_bandit_mc.jl
```

Expected: ~20-30 min (30 seeds × 11 sectors).

- [ ] **Step 5: Create `scripts/04_select_basket.jl`**

```julia
# scripts/04_select_basket.jl
# Pick the median-Sharpe seed from the MC results and write the frozen basket.
# Sharpe proxy here = mean(sector_best_means) since we haven't deployed yet;
# the real Sharpe-on-hold-out lives in the backtest step.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "code"))
using ConstrainedCobbDouglas
using JLD2
using Statistics

const PATH_OUT = joinpath(@__DIR__, "data")

mc = load_results(joinpath(PATH_OUT, "per_sector_bandit_mc_results.jld2"))
seeds = mc["config"]["BANDIT_MC_SEEDS"]
per_seed_means = mc["per_seed_best_means"]
# Score each seed by the average of its per-sector best means
scores = [mean(values(d)) for d in per_seed_means]
order = sortperm(scores)
median_idx = order[ceil(Int, length(order) / 2)]
println("Median-score seed: ", seeds[median_idx], " (score = ", round(scores[median_idx]; digits = 4), ")")
frozen_tickers = mc["per_seed_tickers"][median_idx]
println("Basket: ", join(frozen_tickers, ", "))

out = Dict(
    "tickers"       => collect(frozen_tickers),
    "seed_id"       => seeds[median_idx],
    "sector_quotas" => mc["quotas"],
    "mc_summary"    => Dict(
        "scores_min"    => minimum(scores),
        "scores_median" => median(scores),
        "scores_max"    => maximum(scores),
        "n_seeds"       => length(seeds)))
save_results(joinpath(PATH_OUT, "frozen_basket.jld2"), out)
println("Saved scripts/data/frozen_basket.jld2")
```

- [ ] **Step 6: Run script 04**

```bash
julia --project=code scripts/04_select_basket.jl
```

Expected: <1 s; prints median seed and the 22 tickers.

- [ ] **Step 7: Commit**

```bash
git add scripts/02_train_bandit.jl scripts/03_train_bandit_mc.jl scripts/04_select_basket.jl
git add -f scripts/data/frozen_basket.jld2
git commit -m "scripts/02-04: bandit single-seed, 30-seed MC, median-seed basket pick"
```

---

### Task 24: Script 05 — the 6-strategy bake-off

**Files:**
- Create: `scripts/05_backtest_strategies.jl`

- [ ] **Step 1: Create `scripts/05_backtest_strategies.jl`**

```julia
# scripts/05_backtest_strategies.jl
# Walk all 6 strategies on the 2025-2026 hold-out window. Write
# scripts/data/backtest_results.jld2.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "code"))

using ConstrainedCobbDouglas
using JLD2
using JumpHMM
using Statistics
using Dates

const PATH_INPUTS = joinpath(@__DIR__, "..", "code", "src", "data")
const PATH_OUT    = joinpath(@__DIR__, "data")
const BACKTEST_RNG_SEED = 2026
const B_0 = 100_000.0

println("=" ^ 78)
println("05_backtest_strategies.jl — 6-strategy bake-off")
println("=" ^ 78)

# Load all artifacts
sim_calib    = load_results(joinpath(PATH_OUT, "sim_calibration.jld2"))
basket       = load_results(joinpath(PATH_OUT, "frozen_basket.jld2"))
ohlc_2025    = load_ohlc_jld2(joinpath(PATH_INPUTS,
    "SP500-Daily-OHLC-1-2-2025-to-12-31-2025.jld2"))
ohlc_2026    = load_ohlc_jld2(joinpath(PATH_INPUTS,
    "SP500-Daily-OHLC-1-2-2026-to-04-22-2026.jld2"))

basket_tickers = String.(basket["tickers"])
all_tickers_2025 = ohlc_2025.tickers
col_2025 = Dict(t => i for (i, t) in enumerate(all_tickers_2025))
keep_2025 = [col_2025[t] for t in basket_tickers]
all_tickers_2026 = ohlc_2026.tickers
col_2026 = Dict(t => i for (i, t) in enumerate(all_tickers_2026))
keep_2026 = [col_2026[t] for t in basket_tickers]
prices_2025 = Matrix{Float64}(ohlc_2025.prices[:, keep_2025])
prices_2026 = Matrix{Float64}(ohlc_2026.prices[:, keep_2026])
prices_hold = vcat(prices_2025, prices_2026)
volumes_hold = vcat(Matrix{Float64}(ohlc_2025.volumes[:, keep_2025]),
                    Matrix{Float64}(ohlc_2026.volumes[:, keep_2026]))
dates_hold = vcat(Vector{Date}(ohlc_2025.dates), Vector{Date}(ohlc_2026.dates))
n_days = size(prices_hold, 1)
println("Hold-out: $n_days days, $(length(basket_tickers)) tickers")

# Market prices (SPY or basket EW)
spy_2025_idx = findfirst(==("SPY"), all_tickers_2025)
spy_2026_idx = findfirst(==("SPY"), all_tickers_2026)
if spy_2025_idx !== nothing && spy_2026_idx !== nothing
    market_prices = vcat(Vector{Float64}(ohlc_2025.prices[:, spy_2025_idx]),
                         Vector{Float64}(ohlc_2026.prices[:, spy_2026_idx]))
else
    market_prices = vec(mean(prices_hold; dims = 2))
end

# SIM params slice for basket
sim_calib_tickers = sim_calib["tickers"]
sim_col = Dict(t => i for (i, t) in enumerate(sim_calib_tickers))
αs = Float64[sim_calib["alpha"][sim_col[t]] for t in basket_tickers]
βs = Float64[sim_calib["beta"][sim_col[t]] for t in basket_tickers]
σ_εs = Float64[sim_calib["sigma_eps"][sim_col[t]] for t in basket_tickers]
σ_m = Float64(sim_calib["sigma_market"])

# EWLS init from frozen 2014-2024 OLS estimates
sim_init = Dict(basket_tickers[i] => ewls_init(αs[i], βs[i], σ_εs[i];
    half_life = 252.0, prior_weight = 252.0) for i in eachindex(basket_tickers))

# Load pretrained JumpHMM market surrogate
market_model = load(joinpath(PATH_INPUTS, "pretrained-jumphmm-market-surrogate.jld2"),
    "market_model")

# ADV from sim_calib
adv = Dict(t => Float64(sim_calib["adv"][t]) for t in basket_tickers)

env = (tickers = basket_tickers,
       prices = prices_hold,
       market_prices = market_prices,
       volumes = volumes_hold,
       sim_params_init = sim_init,
       σ_m = σ_m,
       dates = dates_hold,
       market_model = market_model,
       c̄ = 0.05)
cost_model = MyCostModel(commission_per_trade = 0.0, half_spread_bps = 5.0,
                         slippage_κ = 0.001, adv = adv)
tax_rates = (st = 0.37, lt = 0.20)

# The six strategies
spec = MyMPCSpec(z = 1.96, T = 21, N = 1000, D_max = 0.08)
strategies = MyAllocationStrategy[
    EqualWeightStrategy(),
    MinVarBuyHoldStrategy(),
    UnconstrainedCDStrategy(),
    CostAwareMVStrategy(κ = 5.0, c = 0.0005),
    CDWithMPCStrategy(spec = spec),
    ConstrainedCDWithMPCStrategy(spec = spec,
        σ_max = 0.12, K_turnover = 0.10 * B_0, w_max = 0.20)]

println("\nRunning $(length(strategies)) strategies:")
results = compare_strategies(strategies, env, cost_model, tax_rates;
    B₀ = B_0, rng_seed = BACKTEST_RNG_SEED)

println("\nHeadline metrics (after-cost, after-tax):")
println(rpad("Strategy", 35), "  ", rpad("Sharpe", 10),
        rpad("MaxDD%", 10), rpad("Turnover", 10), "n_trig")
for (name, r) in results
    sm = r.summary
    println(rpad(name, 35), "  ",
        rpad(round(sm.ann_sharpe; digits = 3), 10),
        rpad(round(sm.max_drawdown * 100; digits = 1), 10),
        rpad(round(sm.ann_turnover; digits = 3), 10),
        sm.n_mpc_triggers)
end

# Persist
out = Dict(
    "config" => Dict(
        "hold_out_start" => string(dates_hold[1]),
        "hold_out_end" => string(dates_hold[end]),
        "n_days" => n_days,
        "K" => length(basket_tickers),
        "tickers" => basket_tickers,
        "B_0" => B_0,
        "rng_seed" => BACKTEST_RNG_SEED,
        "tax_rates" => Dict("st" => tax_rates.st, "lt" => tax_rates.lt),
        "ewls_half_life_days" => 252),
    "dates"   => dates_hold,
    "results" => results)
save_results(joinpath(PATH_OUT, "backtest_results.jld2"), out)
println("\nSaved scripts/data/backtest_results.jld2")
```

- [ ] **Step 2: Run script 05**

```bash
julia --project=code scripts/05_backtest_strategies.jl
```

Expected: ~5-15 min; prints headline table; writes `backtest_results.jld2`.

- [ ] **Step 3: Commit**

```bash
git add scripts/05_backtest_strategies.jl
git commit -m "scripts/05: 6-strategy bake-off on 2025-2026 hold-out"
```

---

## Phase 11 — Notebook

### Task 25: Create the theory + viewer notebook

**Files:**
- Create: `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb`
- Create: `Include.jl` (at the repo root, loaded by the notebook)

- [ ] **Step 1: Create `Include.jl`**

```julia
# Notebook setup — activate the local package and define data paths.

const _ROOT = pwd()
const _PATH_TO_INPUTS    = joinpath(_ROOT, "code", "src", "data")
const _PATH_TO_ARTIFACTS = joinpath(_ROOT, "scripts", "data")

import Pkg
Pkg.activate(joinpath(_ROOT, "code"))

using ConstrainedCobbDouglas
using JLD2
using DataFrames
using PrettyTables
using Plots
using Statistics
using Dates

function _check_artifact(p)
    if !isfile(p)
        error("Missing artifact: $p\nRun the script that produces it (see scripts/01-05).")
    end
end
```

- [ ] **Step 2: Write the notebook JSON** (`eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb`).

Use `jupyter nbconvert` or the IJulia kernel to create this. For reproducibility, here's the minimal JSON skeleton (write it directly with Write, then open and run cells in Jupyter):

```json
{
 "cells": [
  {"cell_type":"markdown","metadata":{},"source":["# Constrained Cobb-Douglas with MPC — Theory and Hold-Out Results\n",
"\n",
"> **Learning Objectives:**\n",
">\n",
"> - Formulate the constrained Cobb-Douglas allocation problem (budget + covariance + turnover + concentration) and recognize it as a convex program solvable in milliseconds at K=22.\n",
"> - Read the MPC discipline (forward projection band + trigger conditions) as a discipline that converts a clock-driven rebalancer into an event-driven one.\n",
"> - Compare the 6 strategies head-to-head on the 2025-2026 hold-out and interpret what each pairwise difference isolates (constraint effect vs trigger effect).\n"]},
  {"cell_type":"markdown","metadata":{},"source":["## Section 1: Theory Recap\n","\n","The constrained Cobb-Douglas allocator and MPC discipline are described in detail in `docs/superpowers/specs/2026-05-17-constrained-cobb-douglas-design.md`.\n"]},
  {"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":["include(\"Include.jl\")\n"]},
  {"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":["# Load all artifacts\n",
"_check_artifact(joinpath(_PATH_TO_ARTIFACTS, \"sim_calibration.jld2\"))\n",
"_check_artifact(joinpath(_PATH_TO_ARTIFACTS, \"frozen_basket.jld2\"))\n",
"_check_artifact(joinpath(_PATH_TO_ARTIFACTS, \"backtest_results.jld2\"))\n",
"\n",
"sim_calib = load_results(joinpath(_PATH_TO_ARTIFACTS, \"sim_calibration.jld2\"))\n",
"basket    = load_results(joinpath(_PATH_TO_ARTIFACTS, \"frozen_basket.jld2\"))\n",
"bt        = load_results(joinpath(_PATH_TO_ARTIFACTS, \"backtest_results.jld2\"))\n",
"println(\"Basket: \", basket[\"tickers\"])\n",
"println(\"Hold-out: \", bt[\"config\"][\"hold_out_start\"], \" to \", bt[\"config\"][\"hold_out_end\"], \" (\", bt[\"config\"][\"n_days\"], \" days)\")\n"]},
  {"cell_type":"markdown","metadata":{},"source":["## Section 2: Headline Bake-Off (after-cost, after-tax)\n"]},
  {"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":["rows = NamedTuple[]\n",
"for (name, r) in bt[\"results\"]\n",
"    sm = r.summary\n",
"    push!(rows, (Strategy = name,\n",
"        Sharpe = round(sm.ann_sharpe; digits=3),\n",
"        AnnRet_pct = round(sm.ann_return*100; digits=2),\n",
"        MaxDD_pct = round(sm.max_drawdown*100; digits=1),\n",
"        Turnover = round(sm.ann_turnover; digits=3),\n",
"        N_trig = sm.n_mpc_triggers))\n",
"end\n",
"sort!(rows; by = r -> -r.Sharpe)\n",
"pretty_table(DataFrame(rows); backend = :text)\n"]},
  {"cell_type":"markdown","metadata":{},"source":["## Section 3: Wealth Curves\n"]},
  {"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":["p = plot(legend = :outerright, size = (1080, 540),\n",
"          xlabel = \"Trading day\", ylabel = \"Wealth (after-cost, after-tax)\")\n",
"for (name, r) in bt[\"results\"]\n",
"    plot!(p, r.wealth_after_cost_aftertax; label = name, lw = 1.4)\n",
"end\n",
"p\n"]},
  {"cell_type":"markdown","metadata":{},"source":["## Section 4: MPC Trigger Reasons\n"]},
  {"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":["for (name, r) in bt[\"results\"]\n",
"    if !isempty(r.trigger_log)\n",
"        reasons = [t.reason for t in r.trigger_log if t.fired]\n",
"        if !isempty(reasons)\n",
"            counts = Dict(rs => count(==(rs), reasons) for rs in unique(reasons))\n",
"            println(rpad(name, 35), \"  \", counts)\n",
"        end\n",
"    end\n",
"end\n"]},
  {"cell_type":"markdown","metadata":{},"source":["## Disclaimer\n","\n","This content is for educational purposes only and does not constitute investment advice."]}
 ],
 "metadata":{
  "kernelspec":{"display_name":"Julia 1.10","language":"julia","name":"julia-1.10"},
  "language_info":{"name":"julia","version":"1.10"}
 },
 "nbformat":4,"nbformat_minor":5
}
```

- [ ] **Step 3: Verify the notebook loads** (only works if IJulia + Jupyter installed)

```bash
jupyter nbconvert --to notebook --execute \
    eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb \
    --output executed.ipynb
```

Expected: completes without error after a minute or two. If IJulia isn't installed, skip this verification and inspect cells manually.

- [ ] **Step 4: Commit**

```bash
git add Include.jl eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
git commit -m "Add theory+viewer notebook with bake-off table and wealth curves"
```

---

## Phase 12 — Documenter.jl + GitHub Pages

### Task 26: Wire up Documenter.jl docs site

**Files:**
- Create: `docs/Project.toml`
- Create: `docs/make.jl`
- Create: `docs/src/index.md`
- Create: `docs/src/theory.md`
- Create: `docs/src/api/sim.md`
- Create: `docs/src/api/allocator.md`
- Create: `docs/src/api/mpc.md`
- Create: `docs/src/api/costs.md`
- Create: `docs/src/api/tax.md`
- Create: `docs/src/api/bandit.md`
- Create: `docs/src/api/backtest.md`
- Create: `docs/src/usage/pipeline.md`
- Create: `docs/src/usage/notebook.md`

- [ ] **Step 1: Create `docs/Project.toml`**

```toml
[deps]
ConstrainedCobbDouglas = "b2c3d4e5-f6a7-8901-bcde-f12345678902"
Documenter = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
```

- [ ] **Step 2: Create `docs/make.jl`**

```julia
using Pkg
Pkg.develop(path = joinpath(@__DIR__, "..", "code"))
Pkg.instantiate()

using Documenter
using ConstrainedCobbDouglas

makedocs(
    sitename = "Constrained Cobb-Douglas + MPC",
    modules = [ConstrainedCobbDouglas],
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "API" => [
            "SIM"        => "api/sim.md",
            "Allocator"  => "api/allocator.md",
            "MPC"        => "api/mpc.md",
            "Costs"      => "api/costs.md",
            "Tax"        => "api/tax.md",
            "Bandit"     => "api/bandit.md",
            "Backtest"   => "api/backtest.md"],
        "Usage" => [
            "Pipeline" => "usage/pipeline.md",
            "Notebook" => "usage/notebook.md"]],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"))

deploydocs(
    repo = "github.com/varnerlab/modified_cobb_douglas_example.git",
    devbranch = "main")
```

- [ ] **Step 3: Create `docs/src/index.md`**

```markdown
# Constrained Cobb-Douglas + MPC

This package implements a constrained Cobb-Douglas portfolio allocator wrapped
in model-predictive-control (MPC) discipline, with a 6-strategy bake-off
harness on a 2025-2026 hold-out window of the S&P 500.

See the [strategy spec](https://github.com/varnerlab/modified_cobb_douglas_example/blob/main/constrained_cobb_douglas.md)
and the [implementation design](https://github.com/varnerlab/modified_cobb_douglas_example/blob/main/docs/superpowers/specs/2026-05-17-constrained-cobb-douglas-design.md) for context.

## Module map
- [SIM](api/sim.md) — per-ticker SIM calibration + EWLS rolling updates
- [Allocator](api/allocator.md) — constrained CD solver + 4 baselines
- [MPC](api/mpc.md) — forward projection + trigger
- [Costs](api/costs.md) — half-spread + slippage cost model
- [Tax](api/tax.md) — lot-by-lot FIFO ledger
- [Bandit](api/bandit.md) — per-sector ε-greedy universe selection
- [Backtest](api/backtest.md) — strategy-dispatch harness

## Quickstart
- [Pipeline](usage/pipeline.md) — how to run scripts/01-05
- [Notebook](usage/notebook.md) — how to launch the .ipynb against the JLD2 artifacts
```

- [ ] **Step 4: Create `docs/src/theory.md`**

```markdown
# Theory

Restates the math from the implementation design doc, sections 4 (constrained
CD), 5 (MPC), and 7 (bandit). See `docs/superpowers/specs/2026-05-17-constrained-cobb-douglas-design.md`
for the full theory recap; this page links to the API references for each
function that implements a piece of the math.

## Constrained Cobb-Douglas

Objective and constraints — see [`solve_constrained_cd`](api/allocator.md).

## MPC forward projection

Step-by-step under continuous compounding — see [`forward_project`](api/mpc.md).

## Per-sector bandit

Algorithm — see [`train_sector_bandit`](api/bandit.md).
```

- [ ] **Step 5: Create `docs/src/api/sim.md`**

```markdown
# SIM API

```@docs
estimate_sim
build_sim_covariance
compute_market_growth
compute_ema
compute_lambda
compute_preference_weights
ewls_init
ewls_update!
```
```

- [ ] **Step 6: Create `docs/src/api/allocator.md`**

```markdown
# Allocator API

```@docs
solve_constrained_cd
solve_unconstrained_cd_analytical
equal_weight_target
solve_minvar_buyhold
solve_cost_aware_mv
```
```

- [ ] **Step 7: Create the remaining API pages** — same `@docs` pattern, one block per module

`docs/src/api/mpc.md`:
```markdown
# MPC API

```@docs
forward_project
forward_project_closed_form
check_trigger
```
```

`docs/src/api/costs.md`:
```markdown
# Costs API

```@docs
trade_cost
```
```

`docs/src/api/tax.md`:
```markdown
# Tax API

```@docs
open_lot!
close_qty!
summarize_after_tax
```
```

`docs/src/api/bandit.md`:
```markdown
# Bandit API

```@docs
assign_quotas
sector_relative_reward
cd_basket_return
train_sector_bandit
```
```

`docs/src/api/backtest.md`:
```markdown
# Backtest API

```@docs
materialize_orders
should_decide
allocate
run_backtest
compare_strategies
summary_metrics
```
```

- [ ] **Step 8: Create `docs/src/usage/pipeline.md`**

```markdown
# Pipeline

End-to-end:

```bash
julia --project=code scripts/01_calibrate_sim.jl
julia --project=code scripts/02_train_bandit.jl
julia --project=code scripts/03_train_bandit_mc.jl
julia --project=code scripts/04_select_basket.jl
julia --project=code scripts/05_backtest_strategies.jl
```

Each script writes a single JLD2 to `scripts/data/`. The notebook reads
them and renders the headline table + wealth curves.
```

- [ ] **Step 9: Create `docs/src/usage/notebook.md`**

```markdown
# Notebook

Launch the Jupyter notebook from the repo root:

```bash
jupyter notebook eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

The notebook does no compute — it loads JLD2 artifacts produced by
`scripts/01-05` and renders tables + plots.
```

- [ ] **Step 10: Build the docs locally to verify**

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="code"); Pkg.instantiate(); include("docs/make.jl")'
```

Expected: builds to `docs/build/` without errors. (The deploydocs step is a no-op locally.)

- [ ] **Step 11: Commit**

```bash
git add docs/
git commit -m "Documenter.jl docs site: API references, theory, pipeline + notebook guides"
```

---

### Task 27: GitHub Pages deployment workflow

**Files:**
- Create: `.github/workflows/docs.yml`

- [ ] **Step 1: Create the workflow file**

```yaml
name: Documentation

on:
  push:
    branches:
      - main
    tags: ['*']
  pull_request:

jobs:
  build:
    permissions:
      actions: write
      contents: write
      pull-requests: read
      statuses: write
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: '1.10'
      - uses: julia-actions/cache@v2
      - name: Install dependencies
        run: |
          julia --project=docs -e 'using Pkg; Pkg.develop(path="code"); Pkg.instantiate()'
      - name: Build and deploy
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          DOCUMENTER_KEY: ${{ secrets.DOCUMENTER_KEY }}
        run: julia --project=docs docs/make.jl
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/docs.yml
git commit -m "Add GitHub Pages docs deployment workflow"
```

- [ ] **Step 3: Push and confirm Pages is enabled**

After pushing to GitHub, enable Pages in the repo settings (Settings → Pages
→ Source: `gh-pages` branch). The first successful workflow run will create
the `gh-pages` branch and publish.

---

## Self-review checklist (run before handoff)

- [ ] **Spec coverage:** every section of `docs/superpowers/specs/2026-05-17-constrained-cobb-douglas-design.md` has at least one task implementing it.
- [ ] **No placeholders:** all `TBD` / `TODO` / `???` removed.
- [ ] **Type consistency:** field names and method signatures used in later tasks match earlier ones (e.g., `MyConstrainedCDProblem.γ` not `.gamma`; `MyMPCSpec.D_max` consistently).
- [ ] **All `include` lines present:** `Types.jl`, `SIM.jl`, `Allocator.jl`, `Costs.jl`, `Tax.jl`, `MPC.jl`, `Bandit.jl`, `Files.jl`, `Backtest.jl` all uncommented in `ConstrainedCobbDouglas.jl` after their respective tasks.
- [ ] **Exports complete:** every function used in scripts is exported.
- [ ] **All tests run green** at end of each task before commit.

---


