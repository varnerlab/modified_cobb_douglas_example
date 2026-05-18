# S5 Notebook Reformat — Constrained Cobb-Douglas + MPC

**Status:** design / approved through brainstorming.
**Date:** 2026-05-18.
**Target notebook:** `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb`.
**Reference template:** `eCornell-AI-finance-lectures/lectures/session-4/eCornell-AI-Finance-S4-Example-Core-TickerPickerBandit-May-2026.ipynb`.

---

## TL;DR

Reformat the S5 notebook into the four-section layout used by the S4 bandit notebook — **Introduction / Theory Recap / Results / Summary** — and re-run the pipeline at `K_basket = 33` (uniform 3 tickers per GICS sector) so the notebook displays the basket size the user wants. The reformat replaces the current stubbed `## Section 1: Theory Recap` (one-line pointer) with a self-contained recap of the constrained CD optimization, the MPC trigger conditions, the hold-out deployment loop algorithm, and the five baseline allocator equations. The current four `## Section N` top-level headers in Results collapse under a single `## Section 2: Results` with `###` subsections. A new `### Frozen Basket` subsection shows the bandit-selected tickers grouped by sector. A new `## Summary` section adds a headline paragraph, Key Takeaways blockquote, and demoted `### Disclaimer`.

---

## 1. Motivation

The current S5 notebook has four structural problems against the target pattern:

1. **Theory Recap is gutted.** `## Section 1: Theory Recap` contains a single-line pointer to `docs/superpowers/specs/2026-05-17-constrained-cobb-douglas-design.md`. The reader cannot follow the notebook without opening the spec.
2. **Results is fragmented across four top-level headers.** `## Section 2: Headline Bake-Off`, `## Section 3: Wealth Curves`, `## Section 4: MPC Trigger Reasons`, `## Section 5: Multi-Seed Backtest Distribution` are sibling top-level sections rather than subsections of a single Results block.
3. **No Summary.** The notebook ends with `## Disclaimer` at top level; there is no synthesis paragraph or Key Takeaways block.
4. **No visibility into the frozen basket.** The notebook loads `frozen_basket.jld2` but never shows the reader which 22 tickers (in current state) or 33 tickers (after Phase A) make up the basket, or how they distribute across GICS sectors.

The bandit notebook resolves all four through its four-section layout. The reformat mirrors that layout exactly.

---

## 2. Scope

### In scope

- **Pipeline re-run at `K_basket = 33`** with uniform 3-per-sector quotas (Phase A).
- **Notebook structural reformat** to the four-section layout (Phase B).
- **Theory Recap content** drawn from `constrained_cobb_douglas.md` and `code/src/Allocator.jl`. The recap is self-contained — readers do not need to open the spec.
- **A new Frozen Basket roster cell** under Results, showing tickers grouped by GICS sector with the sector quota vector.
- **A new Summary section** with headline paragraph, Key Takeaways blockquote, and demoted Disclaimer.
- **Narrative prose around each Results code cell** explaining what to read.

### Out of scope

- Changes to the constrained CD solver, MPC trigger logic, cost model, or tax model. None of these are touched.
- Changes to scripts `01_calibrate_sim.jl` (universe-independent SIM fit), `02_train_bandit.jl` (dev-only smoke test; output is downstream-orphaned).
- Re-training the SIM. Calibration is per-ticker; basket size changes do not invalidate `sim_calibration.jld2`.
- Adding new strategies, new metrics, new figures beyond what the current notebook already produces.
- Documenter.jl docs site regeneration. (May be triggered separately if needed.)

---

## 3. Phase A — pipeline re-run at K_basket = 33

### 3.1 Single config edit

`scripts/03_train_bandit_mc.jl:14`

```julia
const K_BASKET = 22    # before
const K_BASKET = 33    # after
```

`assign_quotas(sector_groups, K_total)` distributes `K_total ÷ S` per sector with remainder going to the largest sectors first. With `S = 11` GICS sectors and `K_total = 33`, the math is `base = 3, remainder = 0` — every sector gets exactly `q_s = 3`.

### 3.2 Re-run sequence

| # | Script | Output artifact | Notes |
|---|---|---|---|
| 03 | `03_train_bandit_mc.jl` | overwrites `per_sector_bandit_mc_results.jld2` | 30 seeds (1001:1030), per-sector trainer |
| 04 | `04_select_basket.jl` | overwrites `frozen_basket.jld2` | picks median-score seed; `tickers` now length 33 |
| 05 | `05_backtest_strategies.jl` | overwrites `backtest_results.jld2` | single-seed (`BACKTEST_RNG_SEED = 2026`) over 6 strategies |
| 06 | `06_backtest_mc.jl` | overwrites `backtest_mc_results.jld2` | 20-seed (2001:2020) MC backtest over 6 strategies |

`01_calibrate_sim.jl` is **skipped** — SIM fit is per-ticker, universe-independent.
`02_train_bandit.jl` is **skipped** — output `per_sector_bandit_results.jld2` is not consumed by any downstream script (verified: `04_select_basket.jl:12` loads only the MC results from script 03). The stale K=22 artifact on disk is left alone; nothing reads it.

### 3.3 Phase A acceptance criteria

After step 04 completes:

```julia
basket = load_results("scripts/data/frozen_basket.jld2")
@assert length(basket["tickers"]) == 33
@assert all(v == 3 for v in values(basket["sector_quotas"]))
@assert length(basket["sector_quotas"]) == 11
```

After step 06 completes: `backtest_mc_results.jld2` contains `summary` keys for all six strategy names (`EqualWeightStrategy`, `MinVarStrategy`, `UnconstrainedCDStrategy`, `CostAwareMVStrategy`, `CDWithMPCStrategy`, `ConstrainedCDWithMPCStrategy`).

---

## 4. Phase B — notebook reformat

The notebook becomes a four-section document mirroring the bandit notebook's structure.

### 4.1 Target top-level structure

| Position | Header | Cell count |
|---|---|---|
| Introduction | `# Constrained Cobb-Douglas with MPC — Theory and Hold-Out Results` (title) | 1 markdown |
| Theory | `## Section 1: Theory Recap` | 1 markdown |
| Results | `## Section 2: Results` | 1 markdown opener + 5 `###` subsections + nested `####` histogram |
| Summary | `## Summary` | 1 markdown (includes `### Disclaimer`) |

### 4.2 Introduction cell

Single markdown cell. Contains:
- Title (above).
- 1-2 paragraph motivation framing what the notebook answers (the live engine's failure and what constrained CD + MPC fixes).
- Existing **Learning Objectives** blockquote — kept verbatim.
- Closing transition sentence ("Let's walk through the theory and read the bake-off.").

### 4.3 Theory Recap cell (~850 words)

Single markdown cell, structured as below. All math is drawn from `constrained_cobb_douglas.md` §3-§4 (problem formulation, MPC trigger) and `code/src/Allocator.jl` (baseline implementations). No new theory is introduced.

#### 4.3.1 Opening paragraph (~80 words)

Condensed motivation from `constrained_cobb_douglas.md` §1: the live engine ran unconstrained CD at a 30-min clock with the turnover gate disabled and bled cost. The new design adds two disciplined components: a constrained allocator (covariance + turnover + concentration) and an MPC trigger that fires re-allocation only on realized out-of-band events. Universe is the frozen `K = 33`-ticker basket from S4 with uniform `q_s = 3` per GICS sector. Closes with a transition into the math.

#### 4.3.2 Blockquote 1 — Constrained CD optimization (~150 words)

Objective:
$$\max_{n_i \ge 0} \; \sum_{i=1}^{K} \gamma_i \log(n_i)$$

Constraints (displayed as math): budget identity `Σ n_i p_i ≤ B`, covariance budget `wᵀΣw ≤ σ_max²`, turnover budget `c̄ · ‖n − n_prev‖₁ ≤ K_turnover`, concentration cap `w_i ≤ w_max`, non-negativity `n_i ≥ 0`.

One-line definitions of `γ`, `Σ` (SIM-implied), `n_prev`, `c̄`, `K_turnover`. Note that `σ_max` (annualized vol cap, default 12%) and `K_turnover` (default 10% of budget per decision) are the two interpretable knobs.

#### 4.3.3 Blockquote 2 — MPC trigger conditions (~150 words)

In-spec band:
$$\mu_\tau - z\sigma_\tau \le V_\tau \le \mu_\tau + z\sigma_\tau$$

with defaults `z = 1.96`, `T = 21` trading days. Forward projection draws `N = 1000` paths from the SPY-JumpHMM marginal + SIM hybrid (`g_{i,τ} = α_i + β_i g_{m,τ} + ε_{i,τ}`).

Trigger fires when **any** of:
1. `V_τ` exits the band on the realized path.
2. `T` days elapse since last allocation.
3. Realized drawdown from peak exceeds `D_max = 8%` (circuit-breaker).

Between triggers the engine submits no orders. Short paragraph: this is the discipline that fixes the live failure mode.

#### 4.3.4 Algorithm box — hold-out deployment loop

`### Algorithm: Constrained CD with MPC (Hold-Out Deployment Loop)`. Initialize / For each trading day / Output structure mirroring the bandit's `### Algorithm: Per-Sector Sparse Bandit` block:

```
Initialize: basket B (33 tickers), σ_max, K_turnover, z, T, D_max, n_prev = 0.
For each trading day τ in [2025-01-02, 2026-04-22]:
  1. Compute γ_τ from SIM + λ-regime + market growth (no news term).
  2. Forward-project N paths over [τ, τ+T] using SPY-JumpHMM + SIM.
  3. Check trigger conditions: band exit, T-day elapsed, or drawdown > D_max.
  4. If any trigger fires, solve the constrained CD problem for w_target;
     translate to integer shares; apply cost + tax engine to the order set.
  5. Update n_prev, log the trigger reason and turnover consumed.
  6. Otherwise hold; submit no orders.
Output: wealth path V_τ, trigger log, after-cost after-tax summary.
```

#### 4.3.5 Blockquote 3 — Baseline strategy equations (~250 words)

All five baselines use the same frozen 33-ticker basket and identical after-cost / after-tax engine; they differ only in the allocator and the rebalance cadence. Equations are taken verbatim from `code/src/Allocator.jl`:

**(1) EqualWeight (buy-and-hold):** `w_i = 1/K` set once on day 1. (`equal_weight_target`, `Allocator.jl:39`.)

**(2) MinVar (buy-and-hold):**
$$\min_w \; w^\top \Sigma w \quad \text{s.t.} \quad \sum_i w_i = 1, \; w_i \ge 0$$
solved on training-window Σ then held. (`solve_minvar_buyhold`, `Allocator.jl:46`.)

**(3) UnconstrainedCD (daily):** closed-form Cobb-Douglas for preferred names (`γ_i > 0`):
$$n_i = \frac{\gamma_i}{\sum_{j \in \text{pref}} \gamma_j} \cdot \frac{B_{\text{eff}}}{p_i}$$
Non-preferred names (`γ_i ≤ 0`) pin at `n_i = ε = 10⁻³`. Rebalanced every trading day. The live engine's allocator at daily (not 30-min) cadence. (`solve_unconstrained_cd_analytical`, `Allocator.jl:11`.)

**(4) CostAwareMV (daily):**
$$\max_w \; \gamma^\top w - \tfrac{\kappa}{2} w^\top \Sigma w - c \|w - w_{\text{prev}}\|_1 \quad \text{s.t.} \quad \sum_i w_i = 1, \; w_i \ge 0$$
Standard-finance alternative; rebalanced daily. (`solve_cost_aware_mv`, `Allocator.jl:200`.)

**(5) CDWithMPC:** the closed form from (3), invoked only when the MPC trigger fires. Isolates the cadence effect from the constraint effect.

**(6) ConstrainedCDWithMPC (the design):** Blockquote 1's problem, invoked on MPC trigger.

Closing prose: pairwise comparisons isolate effects. **(3) vs (5)** isolates trigger-only; **(5) vs (6)** isolates constraints-only; **(3) vs (6)** is the combined live-engine fix.

#### 4.3.6 Closing paragraph + script links (~40 words)

"The implementation lives in the following scripts (the notebook only loads their saved results):"

- `scripts/01_calibrate_sim.jl` → `sim_calibration.jld2`
- `scripts/04_select_basket.jl` → `frozen_basket.jld2` (with `03_train_bandit_mc.jl` upstream)
- `scripts/05_backtest_strategies.jl` → `backtest_results.jld2`
- `scripts/06_backtest_mc.jl` → `backtest_mc_results.jld2`

### 4.4 Results section — cell-by-cell layout

`## Section 2: Results` is a single top-level header. Subsections use `###` for navigation; the Sharpe-distribution figure uses `####` nested inside Multi-Seed.

| # | Type | Action | Content |
|---|---|---|---|
| 1 | md | NEW | `## Section 2: Results` header + 50-word framing prose + Data-Windows blockquote (see 4.4.1) |
| 2 | code | KEEP | `include("Include.jl")` (unchanged) |
| 3 | code | EDIT | Load three artifacts (`sim_calibration.jld2`, `frozen_basket.jld2`, `backtest_results.jld2`). **Drop** the `println("Basket: ", basket["tickers"])` line. Keep the hold-out window println. |
| 4 | md | NEW | `### Frozen Basket: Tickers and GICS Sectors` + framing prose |
| 5 | code | NEW | Sector roster code (see 4.4.2) |
| 6 | md | EDIT | `### Headline Bake-Off (after-cost, after-tax)` (demoted from `## Section 2:`) + 50-word framing prose pointing at the pairwise comparison rows |
| 7 | code | KEEP | `pretty_table` of the bake-off rows (unchanged) |
| 8 | md | EDIT | `### Wealth Curves` (demoted from `## Section 3:`) + 25-word framing prose |
| 9 | code | KEEP | wealth-curve plot (unchanged) |
| 10 | md | EDIT | `### MPC Trigger Reasons` (demoted from `## Section 4:`) + 50-word framing prose |
| 11 | code | KEEP | trigger-reason print (unchanged) |
| 12 | md | EDIT | `### Multi-Seed Backtest Distribution` (demoted from `## Section 5:`); existing prose mostly preserved |
| 13 | code | KEEP | MC summary table (unchanged) |
| 14 | md | EDIT | `#### Sharpe distribution histogram — ConstrainedCDWithMPCStrategy` (demoted from `###`) + 25-word framing prose |
| 15 | code | KEEP | histogram (unchanged) |

**No code cells are reordered or deleted.** Only markdown headers move and four new pieces appear (Section 2 opener, Frozen Basket subsection header + code, and additional prose around existing code cells).

#### 4.4.1 Section 2 opener — Data Windows blockquote content

> **Data windows:**
>
> - **Training:** 2014-01-03 to 2024-12-31, ~10 years of daily SPY-relative returns for the SIM `(α_i, β_i, σ_{ε,i})` per ticker.
> - **Hold-out:** 2025-01-02 to 2026-04-22, 326 trading days. Every strategy is forward-walked through this window with identical cost + tax rules.
> - **Universe:** 33-ticker basket frozen from the S4 per-sector bandit (median-Sharpe seed from the 30-seed run, uniform `q_s = 3` per GICS sector). The universe does not change during the backtest.

#### 4.4.2 Frozen Basket roster code

```julia
sector_of, _ = load_sector_map(basket["tickers"],
                               joinpath(_PATH_TO_INPUTS, "sp500-sectors.csv"))

roster = DataFrame(
    Ticker = basket["tickers"],
    Sector = [get(sector_of, t, "(unknown)") for t in basket["tickers"]])
sort!(roster, [:Sector, :Ticker])

println("Frozen basket: ", length(basket["tickers"]),
        " tickers, median-Sharpe seed = ", basket["seed_id"])
println("Sector quotas: ", basket["sector_quotas"])
pretty_table(roster; backend = :text)
```

Both `load_sector_map` and `_PATH_TO_INPUTS` are already available from `Include.jl` (`using ConstrainedCobbDouglas` and the const at line 4 respectively). No new infrastructure.

### 4.5 Summary cell

Single markdown cell. Structure:

#### 4.5.1 Headline paragraph (~80 words)

States the design (constrained CD + MPC vs. live engine) and the hold-out result with **numeric placeholders** to be filled from Phase A artifacts:

- `{S_6}` = `median(bt_mc["summary"]["ConstrainedCDWithMPCStrategy"]["sharpe_mc"])`
- `{S_3}` = `bt["results"]["UnconstrainedCDStrategy"].summary.ann_sharpe`
- `{S_1}` = `bt["results"]["EqualWeightStrategy"].summary.ann_sharpe`
- `{T_6}` = ratio of `bt["results"]["UnconstrainedCDStrategy"].summary.ann_turnover` to `bt["results"]["ConstrainedCDWithMPCStrategy"].summary.ann_turnover`

#### 4.5.2 Key Takeaways blockquote (3 bullets, ~250 words total)

**Bullet 1 — Constraints and cadence are both load-bearing.** Pairwise comparison (3→5 trigger-only fix vs 5→6 constraints-only fix) with drawdown-delta placeholders `{ΔDD_53}` = `bt["results"]["UnconstrainedCDStrategy"].summary.max_drawdown - bt["results"]["CDWithMPCStrategy"].summary.max_drawdown` and `{ΔDD_65}` analogously for the second pair.

**Bullet 2 — MPC trigger reasons concentrate on band exits.** Placeholder `{N_trig}` = `bt["results"]["ConstrainedCDWithMPCStrategy"].summary.n_mpc_triggers`. Note that the trigger log dominates the band-exit reason; the calendar refresh and circuit-breaker are rare.

**Bullet 3 — Read distributions, not single trials.** Placeholder `{σ_Sharpe}` = `std(bt_mc["summary"]["ConstrainedCDWithMPCStrategy"]["sharpe_mc"])`. Reminder that only the two MPC strategies are stochastic; the other four collapse to a single point per metric.

#### 4.5.3 Closing paragraph (~50 words)

Closes the loop opened by `constrained_cobb_douglas.md` §1: "no cost, no risk, no holding, no information clock." Constrained CD gives the first three; MPC gives the fourth.

#### 4.5.4 Disclaimer

`### Disclaimer` — demoted from current `## Disclaimer`. ~50 words, adapted from the bandit notebook:

> This content is for educational purposes only and does not constitute investment advice. The examples use real historical data, a frozen SIM calibration on 2014-2024, and a single 2025-2026 forward window; conclusions about cost-aware constrained allocation and MPC trigger discipline do not generalize to other markets, time periods, or client risk profiles without re-calibration.

Trailing `___` horizontal rule matches the bandit notebook's footer.

---

## 5. Phase B acceptance criteria

1. **Structure check:** notebook contains exactly three top-level `##` headers — `## Section 1: Theory Recap`, `## Section 2: Results`, `## Summary`, and no others. The Introduction is the `#` title cell (`# Constrained Cobb-Douglas with MPC — Theory and Hold-Out Results`), not a `##` section.
2. **Subsection check:** under `## Section 2: Results`, the `###` subsections appear in order: Frozen Basket, Headline Bake-Off, Wealth Curves, MPC Trigger Reasons, Multi-Seed Backtest Distribution. The Sharpe-distribution histogram is `####` nested under Multi-Seed.
3. **No top-level Disclaimer.** Disclaimer appears only as `### Disclaimer` under `## Summary`.
4. **Notebook executes end-to-end** with the Phase A artifacts: `jupyter nbconvert --to notebook --execute` (or equivalent) returns exit 0; all cells produce expected output (33-row roster, six-row bake-off table, MC histogram).
5. **All `{...}` placeholders resolved.** No literal `{S_6}`, `{S_3}`, `{T_6}`, `{ΔDD_53}`, `{ΔDD_65}`, `{N_trig}`, `{σ_Sharpe}` strings remain in the committed notebook.
6. **Theory Recap is self-contained.** No "see spec doc X" stubs in `## Section 1: Theory Recap`; the math and algorithm box are inline.

---

## 6. Order of operations

1. Phase A: edit `K_BASKET = 33` in `03_train_bandit_mc.jl`; run `03 → 04 → 05 → 06`; verify Phase A acceptance.
2. Phase B: open `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb`; apply the cell-by-cell edits from §4; resolve all `{...}` placeholders from the freshly-saved artifacts; execute the notebook end-to-end; verify Phase B acceptance.
3. Commit Phase A artifact diffs and Phase B notebook diff as separate commits (Phase A: data; Phase B: notebook + prose).

---

## 7. References

- `constrained_cobb_douglas.md` — full design spec for the constrained CD + MPC strategy (§3 problem, §4 MPC trigger, §5 universe, §7 forward projection).
- `code/src/Allocator.jl` — implementations: `solve_unconstrained_cd_analytical` (line 11), `equal_weight_target` (39), `solve_minvar_buyhold` (46), `solve_constrained_cd` (72), `solve_cost_aware_mv` (200).
- `code/src/Bandit.jl` — `assign_quotas` (line 6); confirms `K = 33`, `S = 11` produces uniform `q_s = 3`.
- `eCornell-AI-finance-lectures/lectures/session-4/eCornell-AI-Finance-S4-Example-Core-TickerPickerBandit-May-2026.ipynb` — the structural template the reformat mirrors.
