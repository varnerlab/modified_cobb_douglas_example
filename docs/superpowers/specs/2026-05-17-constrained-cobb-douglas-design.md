# Constrained Cobb-Douglas with MPC — Implementation Design

**Date:** 2026-05-17
**Status:** approved by user (sections 1–7); ready for writing-plans
**Source spec:** [`constrained_cobb_douglas.md`](../../../constrained_cobb_douglas.md) — the strategy spec; this document covers *how we build it*

---

## TL;DR

Build a self-contained Julia package + script pipeline + Jupyter notebook + Documenter.jl-published API docs that implement the constrained Cobb-Douglas + MPC strategy described in `constrained_cobb_douglas.md`. Strategy math is unchanged; this doc covers the code organization, the data flow between calibration / bandit / backtest scripts, the JuMP conic formulation of the constrained allocator, the SIM-rolling-via-EWLS forward projection that drives MPC, the cost + FIFO tax engines, and the 6-strategy bake-off harness.

---

## 0. Pinned decisions (from brainstorming)

| Topic | Decision |
|---|---|
| Self-containment | Vendor specific functions from `eCornell-AI-finance-lectures/code/src/`; use external packages (JuMP, Clarabel, JumpHMM, JLD2, etc.) as deps in `Project.toml`. No 5000-line `Compute.jl` copy. |
| Bandit basket | Vendor per-sector bandit code; re-train in this repo with 30 seeds (1001-1030); pin median-Sharpe seed; freeze the 22-name basket as a JLD2 artifact. |
| §9 open params from source spec | Pin median-Sharpe bandit seed. σ_max, z/T, N stay at spec defaults; backtest reports sensitivity sweeps. |
| Notebook scope | Theory + viewer; scripts produce all artifacts. Notebook never trains anything (matches S4 bandit notebook pattern). |
| v1 scope | Backtest only. Paper trade is a follow-up doc once backtest results are in hand. |
| Code organization | Modular package: one module per concern (Types, SIM, Allocator, MPC, Costs, Tax, Bandit, Backtest, Files). |
| Docs | Documenter.jl in `docs/`, deployed to GitHub Pages via `.github/workflows/docs.yml`. |
| §6.4 cost validation gate | Dropped — paper-trade data is unreliable, no trustworthy ground truth to calibrate against. |
| Parameter tracking | EWLS-style online SIM updates (vendor `ewls_init` / `ewls_update!` from lectures repo). Frozen 2014-2024 OLS is the *prior*, not the *posterior*. |

---

## 1. Filesystem layout

```
modified_cobb_douglas_example/
├── constrained_cobb_douglas.md          (existing strategy spec, source of truth)
├── README.md
├── .github/workflows/docs.yml           (build + deploy Documenter to gh-pages on push to main)
├── docs/
│   ├── superpowers/specs/               (brainstorming spec output — this file)
│   ├── Project.toml                     (Documenter + DocumenterCitations + local dev-dep on code/)
│   ├── make.jl                          (calls makedocs(...) + deploydocs(repo=".../modified_cobb_douglas_example.git"))
│   └── src/
│       ├── index.md                     (intro, learning goals, links to source spec)
│       ├── theory.md                    (constrained CD math, MPC discipline, forward projection — mirrors notebook §1-§4)
│       ├── api/                         (one page per module; each uses @docs blocks pulling docstrings from code/)
│       │   ├── sim.md
│       │   ├── allocator.md
│       │   ├── mpc.md
│       │   ├── costs.md
│       │   ├── tax.md
│       │   ├── bandit.md
│       │   └── backtest.md
│       └── usage/
│           ├── pipeline.md              (how to run scripts/01-05 end-to-end)
│           └── notebook.md              (how to launch the .ipynb against the JLD2 artifacts)
├── code/
│   ├── Project.toml                     (deps: JuMP, Clarabel, JumpHMM, JLD2, CSV, DataFrames,
│   │                                     Distributions, Statistics, LinearAlgebra, Random, StatsBase)
│   ├── Manifest.toml
│   ├── src/
│   │   ├── ConstrainedCobbDouglas.jl    (umbrella module, exports, package init)
│   │   ├── Types.jl                     (problem/result structs, abstract strategy type + concretes)
│   │   ├── SIM.jl                       (estimate_sim, build_sim_covariance, compute_market_growth,
│   │   │                                 compute_ema, compute_lambda, compute_preference_weights,
│   │   │                                 ewls_init, ewls_update!)
│   │   ├── Allocator.jl                 (solve_constrained_cd via JuMP/Clarabel + 5 baseline allocators)
│   │   ├── MPC.jl                       (forward_project, in-spec band, check_trigger)
│   │   ├── Costs.jl                     (commission/half-spread/slippage; trade_cost; materialize_orders)
│   │   ├── Tax.jl                       (lot-by-lot FIFO ledger, ST/LT classification, summarize_after_tax)
│   │   ├── Bandit.jl                    (per-sector ε-greedy bandit + Monte Carlo driver)
│   │   ├── Backtest.jl                  (allocate, should_decide, run_backtest, compare_strategies)
│   │   ├── Files.jl                     (load/save JLD2 + price data + sector CSV)
│   │   └── data/                        (committed inputs — ship with the repo)
│   │       ├── SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2
│   │       ├── SP500-Daily-OHLC-1-2-2025-to-12-31-2025.jld2
│   │       ├── SP500-Daily-OHLC-1-2-2026-to-04-22-2026.jld2
│   │       ├── sp500-sectors.csv
│   │       └── pretrained-jumphmm-market-surrogate.jld2
│   └── test/
│       ├── runtests.jl
│       ├── test_sim.jl                  (OLS accuracy, covariance PSD, EWLS half-life behavior)
│       ├── test_allocator.jl            (loose-constraint identity, σ_max monotonicity, zero-turnover lock)
│       ├── test_mpc.jl                  (projection moments, trigger conditions)
│       ├── test_costs.jl                (round-trip cost, slippage scaling)
│       ├── test_tax.jl                  (FIFO consumption, ST/LT boundary, partial close)
│       └── test_backtest.jl             (strategy isolation, MPC gating, JLD2 round-trip)
├── scripts/
│   ├── 01_calibrate_sim.jl              (read 2014-2024 OHLC → fit per-ticker SIM + ADV → sim_calibration.jld2)
│   ├── 02_train_bandit.jl               (single seed; per_sector_bandit_results.jld2 — dev sanity check)
│   ├── 03_train_bandit_mc.jl            (30 seeds 1001-1030; per_sector_bandit_mc_results.jld2)
│   ├── 04_select_basket.jl              (median-Sharpe seed → frozen_basket.jld2)
│   ├── 05_backtest_strategies.jl        (all 6 strategies on 2025-2026 → backtest_results.jld2)
│   └── data/                            (artifacts written by scripts; gitignored except frozen_basket.jld2)
│       └── .gitkeep
└── eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
                                          (notebook at repo root; Include.jl activates code/, loads JLD2s)
```

**Notebook conventions:** filename follows the S4 bandit notebook pattern. First cell is an `Include.jl` that runs `Pkg.activate("code")`, `using ConstrainedCobbDouglas`, and defines `_PATH_TO_INPUTS = joinpath(@__DIR__, "code", "src", "data")` and `_PATH_TO_ARTIFACTS = joinpath(@__DIR__, "scripts", "data")`. No compute on the read path. If any artifact is missing the notebook prints a one-liner pointing to the script that produces it.

**Documenter mechanics:** every exported function in `code/src/*.jl` carries a triple-quoted docstring (signature, args, returns, optional usage block — same style as the lectures repo). API pages use `@docs` blocks to pull them in by module. No `@autodocs` over the whole module — the API surface is curated, function-by-function. `usage/*.md` and `theory.md` are hand-written prose.

---

## 2. Types, module interfaces, and strategy abstraction

### 2.1 Strategy dispatch

Six strategies share one harness. Use an abstract type with one concrete struct per strategy, dispatching two methods:

```julia
abstract type MyAllocationStrategy end

struct EqualWeightStrategy <: MyAllocationStrategy end                 # Strategy 1: buy-and-hold EW
struct MinVarBuyHoldStrategy <: MyAllocationStrategy end               # Strategy 2: S1 minvar, no rebalance
struct UnconstrainedCDStrategy <: MyAllocationStrategy end             # Strategy 3: live allocator, daily fire
struct CostAwareMVStrategy <: MyAllocationStrategy                     # Strategy 4: γ + κ·Σ + l1 cost
    κ::Float64
    c::Float64
end
struct CDWithMPCStrategy <: MyAllocationStrategy                       # Strategy 5: unconstrained CD, MPC trigger
    spec::MyMPCSpec
end
struct ConstrainedCDWithMPCStrategy <: MyAllocationStrategy            # Strategy 6: the new design
    spec::MyMPCSpec
    σ_max::Float64
    K_turnover::Float64
    w_max::Float64
end

# Each strategy implements:
allocate(strategy, state, t)       → Vector{Float64}   # target shares
should_decide(strategy, state, t)  → Bool              # is t a decision day?
```

Comparison surface: 5 vs 6 isolates the *constraint* effect; 3 vs 5 isolates the *trigger discipline* effect; 3 vs 6 captures both. Matches source spec §6.2.

### 2.2 Type families

All live in `Types.jl`, exported from `ConstrainedCobbDouglas.jl`:

| Family | Types | Purpose |
|---|---|---|
| Calibration | `MySIMParameterEstimate`, `MyEWLSState` | Vendored from lectures repo. OLS estimate is the initial state; EWLS carries it forward online. |
| Allocator I/O | `MyConstrainedCDProblem`, `MyConstrainedCDResult` | JuMP solver inputs/outputs (see §4) |
| MPC I/O | `MyMPCSpec`, `MyMPCProjection`, `MyMPCTrigger` | Forward projection inputs/outputs + trigger record |
| Cost model | `MyCostModel` | Commission, half-spread, slippage params + ADV table |
| Tax | `MyTaxLot`, `MyTaxLedger` | Lot-by-lot FIFO ledger + closed-lot diagnostics |
| Bandit | `MyBanditConfig`, `MyBanditResult` | Per-sector training config + winning baskets, reward histories, MC distribution |
| Harness | `MyBacktestState`, `MyBacktestResult` | Mutable per-day state + frozen per-strategy output |

### 2.3 Solver result type (corrected)

```julia
struct MyConstrainedCDResult
    n::Vector{Float64}            # optimal continuous shares (no rounding)
    w::Vector{Float64}            # optimal weights
    unallocated_budget::Float64   # nonzero only when no asset has γ > 0
    duals::NamedTuple             # dual values on σ_max, turnover, w_max constraints (when binding)
    status::Symbol                # :optimal, :no_preferred, :infeasible, :solver_failed
    objective::Float64            # achieved Cobb-Douglas log-utility
end
```

**No share rounding in the solver.** Clarabel returns continuous shares. The integer-rounding step lives in `Backtest.jl` as `materialize_orders(n_target, n_current, prices, B_available, cost_model) → (orders::Vector{NamedTuple}, cash_delta::Float64)`, which (a) enforces minimum order size to defuse γ-jitter (live engine failure mode, source spec §1.3), and (b) the turnover budget constraint already bounds total l1 churn at the solver level, so the rounding step rarely faces large discrepancies. Both defenses, not either-or.

### 2.4 Module export contracts

```
SIM.jl       :: estimate_sim, build_sim_covariance, compute_market_growth,
                compute_ema, compute_lambda, compute_preference_weights,
                ewls_init, ewls_update!
Allocator.jl :: solve_constrained_cd, solve_unconstrained_cd_analytical,
                solve_minvar_buyhold, solve_cost_aware_mv, equal_weight_target,
                materialize_orders
MPC.jl       :: forward_project, in_spec_band, check_trigger
Costs.jl     :: build_cost_model, trade_cost
Tax.jl       :: open_lot!, close_qty!, summarize_after_tax
Bandit.jl    :: train_per_sector_bandit, monte_carlo_bandit,
                select_median_seed, assemble_basket
Backtest.jl  :: allocate, should_decide, run_backtest, compare_strategies,
                summary_metrics
Files.jl     :: load_ohlc_jld2, load_sector_map, save_results, load_results
```

`compute_preference_weights` is the no-news variant of the lectures function (source spec §3.3 omits `news_t` / `nu_loadings`).

---

## 3. Data flow and script pipeline

### 3.1 The DAG

```
        code/src/data/                          scripts/data/
        ─────────────                            ─────────────
   ┌─► SP500-OHLC-2014-2024.jld2 ─┐
   │   sp500-sectors.csv          ├─► 01_calibrate_sim.jl ────► sim_calibration.jld2 ─┐
   │                              │   (per-ticker OLS + ADV)                          │
   │                              │                                                   │
   │                              └─► 02_train_bandit.jl ─────► per_sector_bandit_results.jld2
   │                                       │  (single seed=2026; dev sanity check)
   │                                       │
   │                                       └─► 03_train_bandit_mc.jl ──► per_sector_bandit_mc_results.jld2
   │                                                                                  │
   │                                                                                  ▼
   │                                                          04_select_basket.jl ──► frozen_basket.jld2 (committed)
   │                                                                                  │
   ├─► SP500-OHLC-2025.jld2 ──────┐                                                   │
   ├─► SP500-OHLC-2026.jld2 ──────┼─► 05_backtest_strategies.jl ───► backtest_results.jld2
   └─► pretrained-jumphmm-        │   (6 strategies; EWLS rolling SIM)                │
       market-surrogate.jld2 ─────┘                                                   ▼
                                                                                  notebook
```

### 3.2 Inputs (committed in `code/src/data/`)

| File | Source | Purpose |
|---|---|---|
| `SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2` | lectures repo | SIM training window (~413 tickers × 10 years daily) |
| `SP500-Daily-OHLC-1-2-2025-to-12-31-2025.jld2` | lectures repo | Hold-out part 1 |
| `SP500-Daily-OHLC-1-2-2026-to-04-22-2026.jld2` | lectures repo | Hold-out part 2 (326 trading days total) |
| `sp500-sectors.csv` | lectures repo | GICS sector map for bandit |
| `pretrained-jumphmm-market-surrogate.jld2` | lectures repo (`train-market-surrogate.jl`) | SPY-JumpHMM marginal for MPC forward projection (source spec §4.4, §7.1). Vendored as calibration artifact in v1; retraining script deferred to v2. |

### 3.3 Produced artifacts (`scripts/data/`, gitignored except `frozen_basket.jld2`)

| Script | Output | Wall-clock | Contains |
|---|---|---|---|
| `01_calibrate_sim.jl` | `sim_calibration.jld2` | ~10 s | tickers (~413 after universe filter), α/β/σ_ε/r² per ticker, σ_m, per-ticker ADV, universe filter log |
| `02_train_bandit.jl` | `per_sector_bandit_results.jld2` | ~1 min | Single seed=2026 run; sector winners, reward histories, holdout metrics |
| `03_train_bandit_mc.jl` | `per_sector_bandit_mc_results.jld2` | ~20–30 min | 30 seeds (1001–1030); MC distributions for sector-bandit + random-per-sector baselines; per-seed tickers and metrics |
| `04_select_basket.jl` | `frozen_basket.jld2` (committed) | <1 s | Median-Sharpe seed, the 22 frozen tickers, sector quotas, MC summary |
| `05_backtest_strategies.jl` | `backtest_results.jld2` | ~5–15 min | All 6 strategies on 2025-2026; wealth series, MPC trigger log, lot ledger, summary metrics (pre/post-tax, pre/post-cost) |

### 3.4 Reproducibility contract

- Random seeds are explicit script-level constants (`SIM_SEED`, `BANDIT_MC_SEEDS = 1001:1030`, `BACKTEST_RNG_SEED`). No `Random.default_rng()` usage.
- Each script's first action: `versioninfo()` plus a dump of its config block into the JLD2 it writes.
- Universe filter (tickers with full OHLC over the full 2014-2026 span) is computed once in `01_calibrate_sim.jl` and pinned via the ticker list in `sim_calibration.jld2`. Every downstream script reads tickers from there.
- `.gitignore` policy: `scripts/data/*.jld2` ignored by default; exceptions tracked in git are `scripts/data/.gitkeep` and `scripts/data/frozen_basket.jld2` (small, the only artifact needed for headline reproducibility — lets a reader run `05` directly after `01`).

---

## 4. The constrained Cobb-Douglas solver

### 4.1 Mathematical structure

The source-spec §3 problem is convex *only on the preferred set* `S⁺ = {i : γᵢ > 0}` (the `γᵢ·log(nᵢ)` term is convex when `γᵢ < 0`). The lectures' `allocate_cobb_douglas` sidesteps this by pinning non-preferred assets to a minimum share `ε` and optimizing only over the preferred subset. We adopt the same split.

```
max_{n[S⁺] ≥ 0}    Σ_{i ∈ S⁺} γᵢ · log(nᵢ)

s.t.    Σ_{i ∈ S⁺} nᵢ pᵢ                 ≤ B - ε · Σ_{i ∈ S⁻} pᵢ           (budget)
        wᵀ Σ w                            ≤ σ_max²                           (covariance)
        Σ_{i=1..K} |nᵢ - n_prev,i| · c̄    ≤ K_turnover                      (turnover, l1)
        wᵢ                                ≤ w_max,  ∀i                       (concentration)

with    wᵢ = nᵢ pᵢ / B   (and wᵢ = ε·pᵢ/B for i ∈ S⁻ pinned)
```

### 4.2 The JuMP conic formulation

Three cone types do the heavy lifting:

| Constraint | Cone | JuMP form |
|---|---|---|
| `tᵢ ≤ log(nᵢ)` (epigraph of log) | Exponential | `[tᵢ, 1.0, nᵢ] ∈ MOI.ExponentialCone()` |
| `wᵀ Σ w ≤ σ_max²` | Second-order | `[σ_max; Lᵀw] ∈ SecondOrderCone()` where `Σ = L Lᵀ` (Cholesky) |
| `\|nᵢ - n_prev,i\| · c̄ ≤ K_turnover` (l1) | Linear (via slacks) | `uᵢ ≥ \|nᵢ - n_prev,i\|`, then `c̄·Σuᵢ ≤ K_turnover` |

`Σ = L Lᵀ` precomputed per allocation call. `Σ` from source spec §3.4 is PSD by construction; regularize with `Σ + δI`, `δ = 1e-8`, before factorization to handle β-zeros.

Solver: `Clarabel.Optimizer` default; `SCS.Optimizer` fallback. The allocator retries once with SCS if Clarabel returns non-`OPTIMAL` before returning `status = :solver_failed`.

### 4.3 Solver result handling

| `status` | Meaning | Harness response |
|---|---|---|
| `:optimal` | Solver converged | Use `n`, proceed to `materialize_orders` |
| `:no_preferred` | `S⁺ = ∅`; full budget falls out as `unallocated_budget` | Hold cash, skip allocation |
| `:infeasible` | Constraints can't be jointly satisfied | Loosen `K_turnover` 2× and retry once; if still infeasible, fall back to `n_prev` and log |
| `:solver_failed` | Both Clarabel and SCS returned non-optimal | Same fallback |

`duals` exposes which constraint is binding — backtest postmortems can read these to see *why* an allocation landed where it did.

### 4.4 Unit-test contract

| Test | Setup | Assertion |
|---|---|---|
| **Loose-constraint identity** | `σ_max = +∞`, `K_turnover = +∞`, `w_max = 1`, all γ > 0 | `n_constrained ≈ n_unconstrained` (rel tol 1e-6); objective values match |
| **σ_max monotonicity** | Sweep `σ_max ∈ [0.02, 0.50]`; other constraints loose | `wᵀΣw` non-increasing as `σ_max` tightens; at `σ_max = 0.02` `wᵀΣw ≤ σ_max² + ε_tol` |
| **Zero turnover lock** | `K_turnover = 0`, `n_prev` set to known feasible position | `n_constrained ≈ n_prev` exactly |
| **Concentration cap** | `w_max = 0.10`, one γ much larger than others | `max(w) ≤ 0.10 + ε_tol`; budget still satisfied |
| **No-preferred fallback** | All γ ≤ 0 | `status == :no_preferred`; `unallocated_budget ≈ B - ε·Σpᵢ_{S⁻}` |
| **Live-fixture jitter defusal** | Load γ, prices, n_prev from a snapshot of the live engine failure period; `K_turnover = 0.10·B` | `materialize_orders` produces ≤ 3-5 trades (defuses source spec §1.3 jitter) |

### 4.5 Explicit non-goals for v1

- No integer constraint inside the solver. `materialize_orders` rounds.
- No piecewise-linear cost in the objective. Costs enter via the turnover *budget*, not the objective. Strategy 4 (cost-aware MV) is where in-objective cost lives by design — bake-off compares the two approaches.
- No short-selling. `n ≥ 0` is hard.

---

## 5. The MPC layer

### 5.1 Forward projection — under continuous compounding, constant shares

The wealth recursion is derived under the continuous-compounding convention used throughout the codebase, with **constant share counts** between MPC fires (we don't rebalance until the trigger fires again).

**Step 1 — Per-step asset prices** (continuous compounding):

```
Pᵢ,τ,j = Pᵢ,τ-1,j · exp(gᵢ,τ,j · Δt)
```

where `gᵢ,τ,j` is the annualized log growth rate (1/year units) drawn from the SIM under market path `j`, and `Δt = 1/252`.

**Step 2 — Asset paths from SIM** (conditional on the SPY-JumpHMM marginal):

```
G_market[1..T, 1..N] = hmm_simulate(jumphmm_model, T; n_paths = N)

For each path j, each asset i, each step τ:
    g[i,τ,j] = α[i] + β[i] · G_market[τ,j] + N(0, σ_ε[i]²)
```

Idiosyncratic noise drawn fresh per `(i, τ, j)`. Σ-implied cross-asset correlation is carried automatically by the shared `G_market` column (`βᵢβⱼ·σ_m²` off-diagonals).

**Step 3 — Portfolio value** (dollar accounting on constant shares):

```
V[τ,j] = Σᵢ nᵢ · Pᵢ,τ,j
       = Σᵢ nᵢ · Pᵢ,τ-1,j · exp(gᵢ,τ,j · Δt)
```

Equivalently, in weighted form with **drifting** weights `wᵢ,τ-1,j = nᵢ · Pᵢ,τ-1,j / V[τ-1,j]`:

```
V[τ,j] = V[τ-1,j] · (1 + Σᵢ wᵢ,τ-1,j · (exp(gᵢ,τ,j · Δt) - 1))
```

Both forms identical. **Covariance does not appear in the recursion**; it appears through the joint distribution of the `g` draws (Step 2) and surfaces in `σ_τ = std_j V[τ,j]`.

**Step 4 — Band statistics:**

```
μ_τ = mean_j V[τ,j]
σ_τ = std_j V[τ,j]
band(τ) = [μ_τ − z·σ_τ,  μ_τ + z·σ_τ]
```

Defaults: `N = 1000`, `T = 21`, `z = 1.96`.

### 5.2 Closed-form validation path (source spec §7.2)

Alongside the MC projection, compute the lognormal closed-form from SIM moments only. Under constant weights and GBM:

```
μ̃ = wᵀ(α + β·gm̄) - 0.5·wᵀΣw     ← Itô drift correction (covariance enters explicitly)
σ̃² = wᵀΣw                         ← portfolio log-variance per Δt
log(V_T / V_0) ~ N(μ̃ · T · Δt,  σ̃² · T · Δt)
```

`forward_project` returns both bands. The trigger uses the JumpHMM-MC band. The closed-form is logged for sanity; if `|MC_σ_τ - closed_σ_τ| / closed_σ_τ > 0.25` at any τ, set `divergence_warning = true` on the projection result — useful postmortem signal for regime-jump days.

### 5.3 In-spec band + trigger conditions

```julia
function check_trigger(state::MyBacktestState, spec::MyMPCSpec)::MyMPCTrigger
    τ = state.date_idx - state.last_decision_t
    proj = state.last_projection

    # 1. Band exit
    if state.V_t < proj.μ[τ] - spec.z*proj.σ[τ] ||
       state.V_t > proj.μ[τ] + spec.z*proj.σ[τ]
        return MyMPCTrigger(true, :band_exit, τ)
    end

    # 2. Horizon refresh
    if τ >= spec.T
        return MyMPCTrigger(true, :horizon_elapsed, τ)
    end

    # 3. Circuit breaker
    drawdown = (state.wealth_peak - state.V_t) / state.wealth_peak
    if drawdown > spec.D_max
        return MyMPCTrigger(true, :drawdown, τ)
    end

    return MyMPCTrigger(false, :in_spec, τ)
end
```

Between triggers, the harness submits zero orders. The wealth series still marks to market on closing prices, but positions are static. This is the discipline that defuses source spec §1.3 cost drag.

### 5.4 Caching policy

- `forward_project` is the expensive call (`N=1000 × T=21 × K=22`). Done once per trigger fire; cached in `state.last_projection`; reused for daily `check_trigger` calls. At ~10-20 expected fires per 326-day hold-out, that's 10-20 projection calls per strategy, not 326.
- JumpHMM market surrogate JLD2 loaded once at script startup; threaded through `MyMPCSpec.market_model`; never re-read from disk.

### 5.5 Unit test contract

| Test | Setup | Assertion |
|---|---|---|
| **Projection self-consistency** | `w = e_i`; `T = 21`; `N = 5000` | `mean(log V_T/V_0) ≈ (α_i + 0.5·σ_ε,i²)·T·Δt` within MC noise; variance matches `Σ_ii · T · Δt` |
| **Closed-form agreement** | Single-asset, no regime structure | `MC_σ_T / closed_σ_T ∈ [0.85, 1.15]` |
| **Trigger: band exit** | Projection w/ known μ,σ; V outside band | `fired == true`, `reason == :band_exit` |
| **Trigger: horizon** | `date_idx - last_decision_t == spec.T` | `fired == true`, `reason == :horizon_elapsed` |
| **Trigger: drawdown** | wealth peak / V give 9% drawdown w/ `D_max = 8%` | `fired == true`, `reason == :drawdown` |
| **Trigger: in-spec idle** | All conditions slack | `fired == false`; harness submits no orders |

---

## 6. Cost engine and lot-by-lot FIFO tax engine

### 6.1 Cost model

```julia
struct MyCostModel
    commission_per_trade::Float64    # default 0.0 (Alpaca paper / commission-free)
    half_spread_bps::Float64         # default 5.0
    slippage_κ::Float64              # default 0.001 (0.1% slippage at q = ADV)
    adv::Dict{String, Float64}       # per-ticker average daily volume, in shares
end
```

Per-trade cost for a fill of `q_signed` shares at `price` for ticker `t`:

```
|q|              = abs(q_signed)
half_spread_cost = (half_spread_bps · 1e-4) · price · |q|
slippage_cost    = slippage_κ · (|q| / adv[t]) · price · |q|    # quadratic in |q|
commission       = commission_per_trade
total_cost       = half_spread_cost + slippage_cost + commission
```

ADV computed once in `01_calibrate_sim.jl` as `mean(volume_2014_2024)` per ticker, persisted in `sim_calibration.jld2` under key `adv`.

### 6.2 Tax ledger

```julia
struct MyTaxLot
    ticker::String
    open_date::Date
    open_price::Float64
    qty::Int
end

mutable struct MyTaxLedger
    lots::Dict{String, Vector{MyTaxLot}}    # FIFO queue per ticker (front = oldest)
    closed_lots::Vector{NamedTuple}          # (open_date, close_date, qty, st_or_lt, pnl)
    realized_st_pnl::Float64                 # < 365 days
    realized_lt_pnl::Float64                 # ≥ 365 days
end
```

**FIFO consumption algorithm:**

```
close_qty!(ledger, ticker, qty_to_close, price, date):
    remaining = qty_to_close
    while remaining > 0:
        front = ledger.lots[ticker][1]
        take = min(front.qty, remaining)
        holding_days = date - front.open_date
        pnl = take · (price - front.open_price)
        classify (holding_days >= 365) ? :lt : :st
        accumulate pnl into realized_st_pnl or realized_lt_pnl
        push to closed_lots
        if take == front.qty:  popfirst!(ledger.lots[ticker])
        else:                   front.qty -= take
        remaining -= take
    if remaining > 0:  error("close_qty!: more shares closed than open")
```

### 6.3 After-tax summary

```
summarize_after_tax(ledger, rates::(st=0.37, lt=0.20)) → NamedTuple
```

Returns realized ST/LT P&L, taxes (symmetric model: losses generate credits at category rate — uniform bias across all strategies, so relative ranking holds), after-tax realized P&L, `lt_share_of_realized` (tax-efficiency proxy from source spec §6.3), and the full holding-period distribution from `closed_lots`.

**Caveats acknowledged, not modeled in v1:** US tax law's ST→LT loss netting rules and wash-sale rule (source spec §6.5). Both add complexity but uniform bias across the bake-off; the strategy ranking holds. Defer to v2 when modeling client-specific tax situations.

Unrealized terminal P&L is **not** force-closed for the headline after-tax number. Pre-tax and after-tax wealth curves are reported side by side; a separate "hypothetical liquidation tax" number is reported for sensitivity.

### 6.4 Harness integration

Every day in the harness:

```
update prices, mark-to-market, update wealth_peak
if should_decide(strategy, state, date_idx):
    n_target = allocate(...)
    orders   = materialize_orders(n_target, state.positions, prices, cash, cost_model)
    for (ticker, q_signed) in orders:
        cost = trade_cost(cost_model, ticker, q_signed, prices[ticker])
        state.cash -= cost
        if q_signed > 0:
            open_lot!(ledger, ticker, q_signed, prices[ticker], dates[date_idx])
            state.cash -= q_signed · prices[ticker]
        else:
            close_qty!(ledger, ticker, -q_signed, prices[ticker], dates[date_idx])
            state.cash += (-q_signed) · prices[ticker]
        state.positions[ticker] += q_signed
        log trade record
```

Costs deduct from cash at trade time. Taxes accrue lot-by-lot but are **summarized** at end-of-backtest, not deducted from running cash (paid retrospectively at filing time).

### 6.5 Unit test contract

| Test | Setup | Assertion |
|---|---|---|
| **Round-trip half-spread** | Buy 100 @ $100, sell 100 @ $100; ADV→∞ | `total_cost == 2 · (5e-4 · 100 · 100) = $10` |
| **Slippage scales with size²** | Buy 1000 vs 100 shares, same ADV | `slippage(1000)/slippage(100) ≈ 100` |
| **Zero commission default** | Default model, 0 shares | `cost == $0` |
| **FIFO ordering** | Open 100@$50, then 100@$60; close 50@$70 | Closes from $50 lot; pnl = $1000 |
| **Partial close** | Open 100@$50; close 30@$60 | Front lot qty 100→70; pnl = $300 |
| **ST/LT boundary** | Open day 0; close 50 on day 364 vs day 365 | First hits `realized_st_pnl`; second hits `realized_lt_pnl` |
| **Over-close errors** | Open 50, attempt close 60 | `close_qty!` throws |
| **After-tax symmetry** | `realized_st_pnl = +$1000`, `rates.st = 0.37` | `tax_st = $370`, `after_tax_realized_pnl = $630` |

---

## 7. The backtest harness

### 7.1 Single-strategy run loop

`run_backtest(strategy, env, cost_model, tax_rates) → MyBacktestResult`. Walks the hold-out window day by day.

```
initialize state (B_0, empty positions, empty ledger, V_0 = B_0,
                  sim_state seeded with the 2014-2024 OLS estimates from sim_calibration.jld2)

for date_idx = 1..n_days:
    update state.prices from env.price_matrix[date_idx, :]
    state.V_t = Σᵢ state.positions[i]·state.prices[i] + state.cash
    state.wealth_peak = max(state.wealth_peak, state.V_t)

    if should_decide(strategy, state, date_idx):
        # Read CURRENT EWLS-updated SIM params at decision time
        (α_t, β_t, σ_ε,t) = read sim_state for each ticker
        γ_t = compute_preference_weights(...)        # γ at *current* state
        Σ_t = build_sim_covariance(...)              # Σ at *current* state

        n_target = allocate(strategy, state, date_idx)
        orders   = materialize_orders(n_target, state.positions, prices, cash, cost_model)
        execute_orders!(state, orders, cost_model, dates[date_idx])

        if strategy has MPC trigger:
            state.last_projection = forward_project(state, strategy.spec, env)
            state.last_decision_t = date_idx

    if strategy has MPC trigger and not state.just_decided:
        trigger = check_trigger(state, strategy.spec)
        push!(state.trigger_log, trigger)
        if trigger.fired:  state.next_decision_due = true

    # EWLS online parameter update — runs every day regardless of decision
    for ticker in env.tickers:
        g_i_today = (log_return of ticker today) / Δt    # annualized
        g_m_today = (log_return of market today) / Δt
        ewls_update!(state.sim_state[ticker], g_i_today, g_m_today)

    record_step!(state, date_idx)

return build_result(strategy, state, env)
```

The EWLS update runs every day, regardless of whether the strategy decided. All strategies (1–6) share the same SIM state evolution → apples-to-apples comparison preserved. The OLS fit from `sim_calibration.jld2` is the **prior** (day-0 initial state); the EWLS update is the **posterior** evolution through the hold-out.

EWLS half-life: tunable, defaults to ~252 trading days (single-observation influence decays ~50% over a year). Backtest sensitivity sweep alongside σ_max and z.

### 7.2 Per-strategy `should_decide` / `allocate` table

| Strategy | `should_decide` | `allocate` |
|---|---|---|
| `EqualWeightStrategy` | `date_idx == 1` | `ones(K) / K · B / prices` |
| `MinVarBuyHoldStrategy` | `date_idx == 1` | `solve_minvar_buyhold(Σ_initial, bounds)` (uses Σ at day 1, frozen) |
| `UnconstrainedCDStrategy` | every trading day | analytical `allocate_cobb_douglas(γ_t, prices, B)` |
| `CostAwareMVStrategy` | every trading day | JuMP: `max γᵀw - (κ/2)·wᵀΣw - c·\|w-w_prev\|₁` s.t. `Σwᵢ=1, wᵢ≥0` |
| `CDWithMPCStrategy` | `date_idx == 1` OR last trigger fired | analytical unconstrained CD |
| `ConstrainedCDWithMPCStrategy` | `date_idx == 1` OR last trigger fired | `solve_constrained_cd(...)` (the new design) |

### 7.3 Comparison orchestrator

```julia
compare_strategies(strategies, env, cost_model, tax_rates; parallel=false)::Dict{String, MyBacktestResult}
```

Strategies are independent; loop is trivially parallelizable via `Threads.@threads` (`parallel=true`). v1 ships sequential. Each strategy gets its own `MersenneTwister` derived from `BACKTEST_RNG_SEED` for reproducible MPC paths.

### 7.4 `MyBacktestResult`

```julia
struct MyBacktestResult
    strategy_name::String
    strategy_config::NamedTuple
    wealth_after_cost_pretax::Vector{Float64}      # length n_days
    wealth_after_cost_aftertax::Vector{Float64}    # length n_days
    wealth_precost_pretax::Vector{Float64}         # shadow series (gross)
    cash::Vector{Float64}
    positions::Matrix{Float64}                     # n_days × K shares
    trades::Vector{NamedTuple}                     # one per fill
    trigger_log::Vector{MyMPCTrigger}              # empty for non-MPC strategies
    ledger::MyTaxLedger                            # final state with full closed_lots
    summary::NamedTuple                            # metrics from §7.5
end
```

### 7.5 Summary metrics (source spec §6.3)

```
ann_return        = (W_T/W_0)^(252/n_days) - 1                       # after-cost, after-tax
ann_volatility    = sqrt(252) · std(daily_log_returns)
ann_sharpe        = ann_return / ann_volatility
max_drawdown      = max((peak - W_t)/peak)
ann_sharpe_pretax, ann_return_pretax           ← on wealth_after_cost_pretax
ann_sharpe_gross                                ← on wealth_precost_pretax
ann_turnover                = sum(|trade_$|) / mean(W_t) · (252/n_days)
lt_share_of_realized        = realized_lt / (realized_st + realized_lt)
holding_period_median_days  = median(closed_lots.holding_days)
holding_period_q25_q75
n_mpc_triggers              = count(t.fired for t in trigger_log)
trigger_reasons             = (band_exit=, horizon_elapsed=, drawdown=)
n_single_name_dd_15pct_days = ...               # source spec §4.3 failure-mode counter
```

### 7.6 JLD2 schema (notebook contract)

```julia
backtest_results.jld2 = Dict(
    "config"  => Dict(
        "hold_out_start" => Date(2025, 1, 2),
        "hold_out_end"   => Date(2026, 4, 22),
        "n_days"         => 326,
        "K"              => 22,
        "tickers"        => Vector{String},
        "B_0"            => 100_000.0,
        "rng_seed"       => BACKTEST_RNG_SEED,
        "cost_model"     => Dict(...),
        "tax_rates"      => (st=0.37, lt=0.20),
        "ewls_half_life_days" => 252,
        "strategy_params" => Dict(name => params),
    ),
    "dates"   => Vector{Date},                  # length n_days
    "results" => Dict(strategy_name => MyBacktestResult),   # 6 entries
)
```

Notebook reads this and renders four things:

1. **Headline table** — 6 rows × (Sharpe pre/post tax, drawdown, turnover, LT share), sorted by after-tax Sharpe.
2. **Wealth curves** — six lines, after-cost after-tax, with MPC trigger fires marked as vertical ticks.
3. **Trigger reason histogram** — band_exit / horizon_elapsed / drawdown counts for strategies 5 and 6.
4. **Holding-period distribution per strategy** — pulls from `closed_lots`.

### 7.7 Unit test contract

| Test | Setup | Assertion |
|---|---|---|
| **Strategy isolation** | `EqualWeightStrategy`, zero cost, deterministic prices tracking SPY | `wealth_after_cost_pretax` matches independent EW-basket computation within 1e-10 |
| **No-trade days don't accrue cost** | Strategy with no decision on day 5 | trade count at day 5 == day 4 |
| **MPC trigger gates rebalances** | `ConstrainedCDWithMPCStrategy` with synthetic flat market | `n_mpc_triggers == 1 + Σ horizon-elapsed refreshes` |
| **3 vs 5 difference is trigger discipline** | Same γ-generator, same cost model | Strategy 3 has many more trades, higher turnover, lower after-cost Sharpe |
| **EWLS prior == OLS day 0** | Initialize harness; read sim_state at day 0 | `state.sim_state[ticker].α ≈ sim_calibration.alpha[ticker]` exactly |
| **EWLS half-life decay** | Inject a shock observation; check influence after `half_life` days | weight contribution ≈ 0.5 (within numerical tolerance) |
| **Determinism** | Run `compare_strategies` twice with same seed | Results match byte-for-byte (modulo JLD2 timestamps) |
| **JLD2 round-trip** | Save then load `backtest_results.jld2` | All fields recover; ledger.closed_lots length and totals match |

---

## 8. Explicit non-goals for v1

| Item | Why deferred |
|---|---|
| Paper-trade harness for stocks | Spec §8.2 — gets its own design doc once backtest results are in hand. |
| Options overlay | Lives in `options_buildout.md`. Convergence per source spec §10. |
| §6.4 cost-model calibration gate | No trustworthy ground-truth fills from the live engine paper trade. Cost params are set from microstructure assumptions; calibrate-against-real-fills is a v2 task. |
| Walk-forward SIM re-fit beyond EWLS | EWLS *is* our walk-forward mechanism. Full periodic batch re-fit is overkill given EWLS provides continuous parameter updates with a tunable half-life. |
| Multi-currency support | The basket is US-listed S&P 500 equities. No FX exposure. Not relevant to v1. |
| Dividend cash flows | OHLC files are split-adjusted; dividends not modeled. Biases total return slightly low across all strategies uniformly — relative ranking holds. Revisit if a strategy turns on dividend timing. |
| Corporate actions (splits, M&A) | Split-adjusted prices in OHLC are sufficient for this universe and window. |
| Wash-sale rule | Source spec §6.5 — changes timing of loss recognition, not long-run total. Revisit if a strategy generates clustered losses. |
| ST/LT loss-netting subtleties | Symmetric model is uniform-bias across strategies. Revisit alongside wash-sale rule. |
| News as γ-input | Source spec §3.3 explicitly drops it. The live pipeline was overhead; revisit only if strategy underperforms without it. |
| Annual/quarterly bandit retraining | Source spec §5.4 — train-once, freeze. Retraining cadence is a v2 enhancement. |

---

## 9. Reference materials

**Source spec:** `constrained_cobb_douglas.md` (this repo root) — the canonical strategy definition.

**Vendored functions** (copied from `eCornell-AI-finance-lectures/code/src/Compute.jl`):

| Function | Lectures location | Use in this repo |
|---|---|---|
| `estimate_sim` | Compute.jl:96 | One-shot OLS on 2014-2024 (script 01) |
| `build_sim_covariance` | Compute.jl:154 | Σ at each decision time |
| `compute_ema`, `compute_lambda` | Compute.jl:360, 389 | Regime-lens λ at each decision time |
| `compute_market_growth` | Compute.jl:411 | Per-day g_m for SIM regression and forward projection |
| `compute_preference_weights` | Compute.jl:459 | γ_t at each decision time (no-news variant) |
| `allocate_cobb_douglas` | Compute.jl:505 | Strategy 3 and 5 allocator (analytical unconstrained CD) |
| `ewls_init`, `ewls_update!` | Compute.jl:3108, 3151 | Online SIM parameter updating through the hold-out |
| Per-sector bandit code | `lectures/session-4/scripts/bandit/per_sector_bandit.jl` | Vendored into `code/src/Bandit.jl` + `scripts/02/03/04` |

**External packages** (deps in `code/Project.toml`, not vendored):

JuMP, Clarabel, SCS, JumpHMM, JLD2, CSV, DataFrames, Distributions, Statistics, LinearAlgebra, Random, StatsBase.

**Pre-trained calibration artifacts** (vendored as binary data, not re-trained in v1):

- `pretrained-jumphmm-market-surrogate.jld2` — produced by `lectures/code/scripts/train-market-surrogate.jl`; consumed by `MPC.jl::forward_project`.

---

## Appendix: Disclaimer

This is a design document for a real-money trading strategy. The constrained Cobb-Douglas + MPC framework is not a guaranteed-return product; all risk-of-loss caveats from the source spec apply. The strategy is backtested first, paper-traded next, and only deployed with client capital after both phases produce satisfactory results.
