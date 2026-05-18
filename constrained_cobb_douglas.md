# Constrained Cobb-Douglas with Model-Predictive Control

**Status:** brainstorm / design doc. Not yet implemented.
**Date:** 2026-05-17
**Sibling doc:** `options_buildout.md` (parallel options-infrastructure track that this strategy will eventually fold into).

---

## TL;DR

Replace the live intraday Cobb-Douglas (CD) engine with a **constrained CD allocator** wrapped in **model-predictive control (MPC)** discipline. The objective is unchanged — the investor maximizes Cobb-Douglas utility over a curated basket — but allocation now respects a covariance budget and a turnover budget, and the engine fires only when the realized portfolio path leaves a forward-projected confidence band, not on a 30-min clock. Universe is selected once by the per-sector ε-greedy bandit from Session 4 (trained on 2014-2024, frozen on a chosen seed). Backtested on 2025-2026 hold-out against five baselines, then paper-traded.

---

## 1. Motivation: what the live engine failed at

The live engine ran from May 5-15, 2026 on a paper account. It was discontinued at market close on 2026-05-18. The diagnostic record is in conversation history; the structural findings that motivate this design are:

1. **Cobb-Douglas was solved unconstrained at every 30-min fire.** Each fire re-derives `n_i ∝ γ_i / p_i` from the just-updated SIM parameters and λ, then rebalances. There is no covariance term, no transaction-cost term, no prior-position term in the objective.
2. **The turnover compliance gate was disabled.** `turnover_limit = 2.0` (200% per fire) was set as a temporary measure to absorb the S1→engine convergence and never tightened back. The gate that should have stopped intraday jitter from churning the book was inactive.
3. **γ jitter at 30-min cadence was rounded into orders.** Sub-percent shifts in γ shifted target shares enough that `round(target_shares − current_shares)` produced ±1-share orders on 5-10 tickers per fire. ~180 auto-submitted orders/day at ~$70 notional each. The cost drag matches the observed flat-with-bleed P&L (−$74 / 9 days on $100K).
4. **η, the policy adaptive parameter, was shadow-logged only.** The live allocator did not consume it. It is not part of this design except as a reference signal we may revive later.
5. **News was a routing-only signal.** Severity gated trades to a human review queue; it did not enter γ. The pipeline consumed ~110 API searches/day to produce ~5 hourly news writes/day, hit rate limits routinely, and exhausted credit balance partway through 2026-05-15. The new design treats news as out-of-scope for γ.
6. **Universe was S1 minvar (22 tickers).** The per-sector bandit existed as a trained artifact but was never wired into production. This design promotes it.

The single sentence: **the engine had no notion of cost, no notion of risk, no notion of holding, and it fired on a clock that had nothing to do with information arrival.**

---

## 2. Concept: constrained CD as MPC

The decision system has three layers, each with a clean interface:

- **Universe layer.** Picks the basket once, before backtest/deployment. Output: a list of 22 tickers from S&P 500 with GICS-sector quotas.
- **Allocation layer.** Given a basket and a current state, solves the constrained CD problem for target weights. Output: `w_target ∈ ℝ^22`.
- **MPC layer.** At time `t`, projects the portfolio forward `T` days under SIM dynamics. The current allocation is "in spec" as long as realized portfolio value `V_τ` stays within a confidence band around the projection. When `V_τ` exits the band, the allocation layer is re-invoked with the latest state.

Between MPC re-triggers, the engine **does nothing**. No fires, no orders, no API calls. Cadence is event-driven by realized state, not by a 30-min cron.

---

## 3. Decision model: the constrained CD optimization

### 3.1 The objective

$$\max_{n_i \ge 0}\; \sum_{i=1}^{K} \gamma_i \log(n_i)$$

This is the log-utility form of Cobb-Douglas. The objective is concave in `n` for `γ_i > 0` and recovers the canonical microeconomic preference story: `γ_i` is the preference exponent for asset `i`, and the optimum allocates budget shares proportional to preferences.

### 3.2 The constraints

$$\sum_{i=1}^{K} n_i p_i = B \quad \text{(budget identity)}$$

$$w^\top \Sigma w \le \sigma_{\max}^2 \quad \text{(covariance budget)}$$

$$\| n - n_{\text{prev}} \|_1 \cdot \bar{c} \le K_{\text{turnover}} \quad \text{(turnover budget)}$$

$$w_i \le w_{\max} \quad \text{(concentration cap per name)}$$

$$n_i \ge 0 \quad \text{(long-only, after the day-2 short-selling fix)}$$

where `w_i = n_i p_i / B`, `Σ` is the SIM-implied covariance (eq. 3.4), `n_{prev}` is the position from the prior MPC decision, `c̄` is the average per-share cost (commission + half-spread + slippage estimate), and `K_{turnover}` is a dollar turnover budget per decision.

### 3.3 The preference vector γ

Computed from SIM parameters per ticker without a news term:

$$\gamma_i = \tanh\!\left( \frac{\alpha_i}{|\beta_i|^\lambda} + |\beta_i|^{1-\lambda} \cdot g_m \right)$$

`α_i, β_i, σ_{ε,i}` come from the SIM regression trained on 2014-2024 daily returns. `λ` is the regime-lens parameter from EMA-crossover on the market index. `g_m` is the smoothed market growth rate. This is unchanged from the live engine except for the **omission of the news term**: `news_t` and `nu_loadings` are not passed.

### 3.4 The covariance matrix Σ

SIM-implied, decomposed:

$$\Sigma = \sigma_m^2 \, \beta \beta^\top + \mathrm{diag}(\sigma_{\varepsilon,1}^2, \ldots, \sigma_{\varepsilon,K}^2)$$

`σ_m^2` is the annualized variance of the market index. This decomposition is the same one used to fit the SIM. No separate sample-covariance estimate is needed; `Σ` is a function of the same parameters that drive `γ`.

### 3.5 Solver

The constrained CD problem is convex (concave objective, linear/convex-quadratic constraints, polyhedral feasible region). For `K ≈ 22` it solves in milliseconds via JuMP with Clarabel or SCS. No closed-form solution — that's the cost of buying covariance and turnover awareness.

### 3.6 The two scalar parameters that drive behavior

- **`σ_max`** sets risk tolerance. Annualized portfolio volatility cap. Default: 12% (a typical balanced-portfolio number; replace with client-specific value before paper trade). Calibrated to client risk profile, not to backtest.
- **`K_turnover`** sets the per-decision turnover budget. The MPC trigger fires re-allocation; `K_turnover` controls how much the new allocation is allowed to differ from the prior. Default: 10% of `B` per re-allocation (an absolute cap; the optimizer chooses to use less if `γ` hasn't moved enough). Calibrated by backtest sensitivity.

Both are interpretable in the client conversation. Neither is a κ-style risk-aversion coefficient with no natural units.

---

## 4. MPC discipline: when to re-allocate

### 4.1 In-spec definition (portfolio-level)

At MPC decision time `t`, compute a forward projection of portfolio value `V_τ` for `τ = t+1, t+2, \ldots, t+T`. The projection produces a mean `μ_τ` and a standard deviation `σ_τ` at each horizon. The current allocation is **in spec** at time `τ` if and only if:

$$\mu_\tau - z \cdot \sigma_\tau \le V_\tau \le \mu_\tau + z \cdot \sigma_\tau$$

The defaults: `z = 1.96` (the 95% confidence band), `T = 21 trading days` (a typical monthly horizon). Both tunable in backtest.

### 4.2 Trigger logic

The MPC layer fires re-allocation when **any one** of the following holds:

1. `V_τ` exits the confidence band on the realized path.
2. `T` days have elapsed since the last allocation (the projection horizon has been reached; refresh).
3. A circuit-breaker condition trips: realized drawdown from peak exceeds `D_max` (default 8%).

Between triggers, **the engine submits no orders**. This is the discipline that fixes the live failure mode.

### 4.3 Known failure mode (open risk)

A single name decays badly while the portfolio band holds → the position is not acted on. The honest move is to **not pre-build a per-name override**. Run the portfolio-only spec in backtest; if the failure mode shows up empirically, add a per-name drawdown stop as a guardrail. Designing for hypothetical asymmetries before observing them is the failure of the live engine in reverse.

### 4.4 Forward projection model (Section 7 has the math)

SPY-JumpHMM marginal for the market path + SIM for asset paths conditional on the market. Cross-asset correlation is carried by the shared market factor through `β`. Idiosyncratic returns remain independent across names. This is the same model that produces `γ` and `Σ`; using it for forward projection is consistency, not new modeling.

---

## 5. Universe selection: the per-sector bandit

### 5.1 What it is

The per-sector ε-greedy combinatorial bandit from S4 (`scripts/bandit/per_sector_bandit.jl` in `eCornell-AI-finance-lectures`). Eleven parallel bandits, one per GICS sector. Each bandit picks `q_s` tickers from its sector's candidate set. Quotas `(q_1, \ldots, q_{11})` sum to `K_{\text{basket}} = 22`.

### 5.2 Reward signal

Sector-relative 21-day forward log return of the Cobb-Douglas-allocated sub-basket:

$$R_s(\mathcal{B}, d) = r_{\mathcal{B}}(d, d+21) - r_{\mathrm{EW},s}(d, d+21)$$

`EW` is equal-weight within the sector. Subtracting the sector equal-weight return strips out the within-sector market factor that the bandit didn't choose — what's left is the basket-selection signal.

### 5.3 Training and freezing

- Trained on 2014-01-03 to 2024-12-31 (~10 years of S&P 500 daily closes).
- Universe filter: tickers with full OHLC coverage on every training and hold-out day. ~413 tickers after the filter.
- 30 seeds (1001-1030) trained in S4. The median-Sharpe seed across the 30 is selected as the canonical basket for the new design. (Pin the seed; do not re-train.)
- The selected basket is frozen for the entire 2025-2026 backtest hold-out window and for the initial paper trade.
- The S4 result: median bandit beats 4 of 5 Claude-curated archetype baskets on hold-out Sharpe and ties the 5th. Random-per-sector (same quotas, no learning) beats 2 of 5. The learning adds incremental edge over the quotas alone.

### 5.4 Retraining cadence (deferred decision)

The spec defaults to **train-once, freeze**. Periodic re-training (annual or quarterly) is a future enhancement; the spec notes this as out of scope for v1.

---

## 6. Backtest design

### 6.1 Train/test split

- **SIM warm-up:** 2014-01-03 to 2024-12-31 (10 years). Fits per-ticker `(α, β, σ_ε)` from daily SPY-relative returns.
- **Hold-out:** 2025-01-02 to 2026-04-22 (326 trading days). All metrics are computed on this window. No parameter or hyperparameter fitting on hold-out data.

### 6.2 Strategies under comparison

Five plus passive:

1. **Equal-weight passive** (deepest baseline; buy-and-hold the 22-name basket equal-weighted).
2. **Min-var buy-and-hold** (S1 baseline; minimum-variance portfolio on training-window Σ, no rebalance).
3. **Unconstrained CD with daily rebalance** (live's allocator, but daily instead of 30-min — isolates the "myopic CD with no constraints" effect from the "30-min cadence" effect).
4. **Cost-aware MV with γ** (`max γ^T w - (κ/2) w^T Σ w - c \|w - w_{prev}\|_1`; the standard-finance alternative to constrained CD).
5. **CD with MPC** (unconstrained CD allocation, but only re-allocate on out-of-spec trigger; isolates the MPC-trigger effect from the constraint effect).
6. **Constrained CD with MPC** (the new design).

Cross-strategy comparison isolates which design choice matters: constraints (5 vs 6), trigger discipline (3 vs 5), or both (3 vs 6).

### 6.3 Metrics

All metrics reported **net of cost and net of tax**:

- Annualized Sharpe ratio (after-cost, after-tax).
- Maximum drawdown.
- Annualized turnover (`Σ |Δ$| / portfolio_value`).
- Holding-period distribution per closed position (the tax-efficiency proxy — fraction of $-realized in long-term-eligible lots).
- Number of MPC re-trigger events per year (for strategies 5 and 6).
- Number of out-of-band days the trigger missed (the failure-mode counter from §4.3).

Reported alongside the same metrics on a **pre-tax pre-cost** basis for sensitivity.

### 6.4 Cost model

Must reproduce the live failure mode (flat-with-bleed under unconstrained CD + 30-min cadence). The model:

- **Commission:** `$0` (matches Alpaca paper / commission-free brokers).
- **Half-spread:** 5 bps per side on equities (10 bps round-trip). Tightened to 2 bps on liquid mega-caps if needed.
- **Slippage:** linear in order size relative to ADV. For order of size `q` shares with average daily volume `ADV`: `slippage = α · (q / ADV) · price`, default `α = 0.1` (basis points per percent-of-ADV). For typical orders this is sub-bp; for large orders it matters.
- **Bid-ask cost:** modeled as half-spread, so reflected in the 5-bps line.

**Validation gate:** if running strategy 3 (unconstrained CD + daily) at 30-min cadence does not reproduce the live flat-with-bleed P&L (~0 ± 50 bps over 9 days), the cost model is wrong. Tune before trusting any other result.

### 6.5 Tax model

Lot-by-lot, FIFO close ordering:

- Each open creates a tax lot `(open_date, open_price, qty)`.
- Each close consumes lots in FIFO order.
- Holding period `< 365 days` → short-term gain/loss; `≥ 365 days` → long-term.
- Apply blended federal rates: `37%` ST, `20%` LT (defaults; client-specific overrides supported).
- **Wash-sale rule is not modeled in v1.** It changes the timing of loss recognition, not the long-run total; revisit if the strategy generates clustered losses around realized winners.

Report **pre-tax** and **after-tax** P&L side by side. After-tax is the number that matters to the client.

### 6.6 Backtest harness

A single Julia driver:

1. Load training-window data, fit SIM, compute `Σ` and seed `γ`-generators.
2. Load the bandit-selected basket (pinned seed from `per_sector_bandit_results.jld2`).
3. For each of the 6 strategies, walk the hold-out window:
   - Strategy-specific allocation logic at decision times.
   - Cost + tax engine applied to every order.
   - State persistence per decision (lot history, current positions).
4. Emit a results JLD2 + comparison figures + summary tables.

---

## 7. Forward projection model

### 7.1 Pipeline (the hybrid SPY-JumpHMM + SIM)

For each forward-projection call at decision time `t`, with horizon `T`:

1. **SPY paths.** Draw `N` paths of the market index from the JumpHMM marginal calibrated on the training window. Each path is a length-`T` vector of daily log returns `g_m^{(j)} = (g_{m,1}^{(j)}, \ldots, g_{m,T}^{(j)})`.

2. **Asset paths.** For each path `j` and each asset `i`:

   $$g_{i,\tau}^{(j)} = \alpha_i + \beta_i \cdot g_{m,\tau}^{(j)} + \varepsilon_{i,\tau}^{(j)}, \quad \varepsilon_{i,\tau}^{(j)} \sim \mathcal{N}(0, \sigma_{\varepsilon,i}^2)$$

   Cross-asset coupling is carried by `β` via the shared `g_m^{(j)}`. Idiosyncratic terms are independent across `i` and across `τ`.

3. **Portfolio paths.** For current weights `w`:

   $$V_\tau^{(j)} = V_0 \cdot \prod_{s=1}^{\tau} \left( 1 + \sum_i w_i \cdot \left( e^{g_{i,s}^{(j)}} - 1 \right) \right)$$

   Wealth dynamics are continuously compounded within day, discretely compounded across days.

4. **Band statistics.** `μ_τ = mean_j V_τ^{(j)}`, `σ_τ = std_j V_τ^{(j)}`. The in-spec band is `[μ_τ − z σ_τ, μ_τ + z σ_τ]`.

### 7.2 Why not closed-form

A lognormal closed-form using only the SIM moments (`μ_p = w^T α`, `σ_p^2 = w^T Σ w`) is cheaper and adequate for a first-pass sanity check. The spec includes it as a **validation path** — at decision time `t`, both the closed-form and the JumpHMM-MC projections are computed, and large divergences (e.g., the JumpHMM tails are dramatically heavier than lognormal) are logged for inspection. The trigger decision uses the JumpHMM-MC band.

### 7.3 Upgrade path (out of scope for v1)

The full multivariate story is the JumpHMM copula in `heston_implied_volatility_model`'s `pretrained-portfolio-surrogate.jld2`. We do not pull this in for v1; the SPY-marginal + SIM hybrid is sufficient and consistent with how `γ` and `Σ` are produced.

---

## 8. Implementation plan

### 8.1 Scope of v1

Five tracks, sequenced:

1. **New repo bootstrap.** Clone of this repo's layout, but with `Compute.jl`-equivalent rebuilt to support the constrained CD solver and the SPY-JumpHMM + SIM forward projection. The S4 lectures repo is referenced for SIM-calibration code only; not vendored as a dependency.
2. **SIM calibration on 2014-2024.** Per-ticker `(α, β, σ_ε)` produced as a JLD2 artifact and committed. Universe is the bandit-selected 22 names.
3. **Constrained CD solver.** JuMP + Clarabel formulation of §3. Unit-tested against unconstrained CD (the constraint slacks should bind progressively as `σ_max` tightens).
4. **MPC harness.** Forward projection (§7), in-spec band, trigger logic, lot-by-lot order book with cost and tax engines.
5. **Backtest comparison.** All 6 strategies on the 2025-2026 hold-out window. Tables, figures, after-tax summary.

### 8.2 Out of scope for this doc

- Phase B / paper-trading harness for stocks → its own design doc once backtest results are in hand.
- Options overlay → `options_buildout.md`. The two tracks rejoin per §10.

### 8.3 Estimated effort

Two to four weeks of focused work for one developer (the user). The solver and projection are small; the cost+tax engine and the rigorous baseline comparison are most of the work.

---

## 9. Open questions

1. **σ_max target.** 12% annualized is a sensible default for a balanced portfolio. The actual client risk tolerance should override before paper trading. Resolution: client conversation, not data.
2. **Z-score for the in-spec band.** Default `z = 1.96` (95%). Tighter `z` (e.g., 1.28 = 80%) re-triggers more often and reacts faster but produces more turnover. Backtest sensitivity to `z` on hold-out.
3. **T (projection horizon).** Default 21 trading days. Aligns with the bandit's reward horizon and the typical credit-spread DTE. Backtest sensitivity.
4. **Number of MC paths in forward projection.** Default `N = 1000`. Cheaper at decision time than the 1000-path heston scenarios because we only project 21 days and we don't need per-contract variance evolution.
5. **Bandit seed selection.** Defaults to median-Sharpe seed across the 30 trained seeds. Alternative: average across seeds (basket is the union of all 30 selected baskets, weighted by selection frequency). The first is reproducible; the second is more honest about model uncertainty. Pick before backtest starts.
6. **Wash-sale rule.** Currently out of scope. Revisit if the strategy generates clustered losses; otherwise, leave it.
7. **News as a future γ-input.** The `compute_preference_weights` function in the lectures repo supports `news_t, nu_loadings`. We are not using it. If the strategy works without news, this confirms the live news pipeline was overhead. If the strategy underperforms, news re-enters scope.

---

## 10. Convergence with `options_buildout.md`

When both tracks (this one and the options buildout) are independently validated in paper trading, the merged system is:

- **Stock sleeve:** Constrained CD with MPC, on the bandit-selected 22-name basket. Capital share `w_stock` (default 0.60-0.75).
- **Options sleeve:** Bull put credit spreads on high-γ names, bear call credit spreads on low-γ names, sized from the same `γ` vector that drives stock allocation. Capital share `w_premium` (default 0.25-0.40).
- **Cash buffer:** 5-10%.

The combined system is the strategy described in `long_short_portfolio.md`, but with the live-failure modes fixed: cost-aware constrained allocator, event-driven cadence, bandit-selected defensible universe, lot-by-lot tax accounting.

---

## 11. References

**Allocators**
- Cobb, C. W. and Douglas, P. H. (1928). "A Theory of Production." *American Economic Review*. — Original Cobb-Douglas utility.
- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*. — Mean-variance baseline (strategy 4 in §6.2).
- Sharpe, W. F. (1963). "A Simplified Model for Portfolio Analysis." *Management Science*. — Single-index model that produces `(α, β, σ_ε)` and the SIM-implied Σ.

**MPC**
- Camacho, E. F. and Bordons, C. *Model Predictive Control*. Springer. — Classical MPC reference; horizon, projection, recourse.
- Boyd, S. *et al.* (2017). "Multi-Period Trading via Convex Optimization." *Foundations and Trends in Optimization*. — Closest thing in finance to what this design proposes.

**Bandit**
- Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem." *Machine Learning*. — ε-greedy analysis.
- The S4 notebook `eCornell-AI-Finance-S4-Example-Core-TickerPickerBandit-May-2026.ipynb` for the per-sector decomposition specific to this problem.

**Live engine reference**
- `eCornell-AI-finance-lectures/lectures/session-4/scripts/production_runner.jl` (for the live engine being replaced).
- `eCornell-AI-finance-lectures/code/src/Compute.jl` (for SIM, EWLS, λ, γ, classify_regime).

---

## Appendix: Disclaimer

This document is a design proposal for a real-money trading strategy. The constrained Cobb-Douglas + MPC framework is not a guaranteed-return product; all the risk-of-loss caveats from the live engine still apply. The strategy will be backtested first, paper-traded next, and only deployed with client capital after both phases produce satisfactory results to the satisfaction of the client and the author. The author is not a licensed investment professional; readers consulting this document for their own use should consult their own risk tolerance and a licensed advisor.
