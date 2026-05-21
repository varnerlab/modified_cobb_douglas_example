# Lambda Swap Note — Sigmoid → Signed Gain-Based Form

**Date:** 2026-05-21
**Files touched:** `code/src/SIM.jl`, `code/test/test_sim.jl`
**Reason for note:** patch must be applied to a second running engine that shares this code path.

## The Issue

`compute_lambda` in `code/src/SIM.jl` was vendored from `Compute.jl` as an **unsigned sigmoid** crossover:

```julia
λ_t = 1 / (1 + exp(-(short_ema_t - long_ema_t) / θ))    # OLD — bounded in [0, 1]
```

This disagrees with the lecture-materials definition of the regime-lens, which is **signed** and gain-based:

```julia
λ_t = -G · (short_ema_t / long_ema_t - 1)               # NEW — signed, unbounded
```

Sign convention in the signed form:

| λ_t       | Regime  | Meaning                                       |
|-----------|---------|-----------------------------------------------|
| λ_t > 0   | bearish | short EMA below long EMA; engine risk-averse  |
| λ_t < 0   | bullish | short EMA above long EMA; engine takes risk   |
| λ_t ≈ 0   | neutral | no strong directional signal                  |

The signed convention is load-bearing — the sign carries information that the sigmoid form discards (sigmoid maps both regimes onto $[0, 1]$, losing the bear/bull axis). Downstream allocators that interpret bearish vs bullish need the sign back.

## The Change (code/src/SIM.jl)

Replaced the body and docstring of `compute_lambda`. Keyword renamed `θ → G` (gain). Signature is otherwise compatible: positional args unchanged, all existing call sites use the default and continue to work.

**Before** (`code/src/SIM.jl:86-98`):
```julia
"""
    compute_lambda(short_ema, long_ema; θ = 0.5) -> Vector{Float64}

Regime-lens λ from EMA crossover. λ_t = 1 / (1 + exp(-(short-long)/θ)) (sigmoid).
"""
function compute_lambda(short_ema::Vector{Float64}, long_ema::Vector{Float64};
        θ::Float64 = 0.5)::Vector{Float64}
    @assert length(short_ema) == length(long_ema)
    diff = (short_ema .- long_ema) ./ θ
    return 1.0 ./ (1.0 .+ exp.(-diff))
end
```

**After:**
```julia
"""
    compute_lambda(short_ema, long_ema; G = 1.0) -> Vector{Float64}

Signed regime-lens λ from an EMA crossover sentiment signal:

    λ_t = -G · (short_ema_t / long_ema_t - 1)

with gain G > 0. ...
"""
function compute_lambda(short_ema::Vector{Float64}, long_ema::Vector{Float64};
        G::Float64 = 1.0)::Vector{Float64}
    @assert length(short_ema) == length(long_ema)
    return -G .* (short_ema ./ long_ema .- 1.0)
end
```

## Downstream Consumers — What to Check

`compute_lambda` feeds `compute_preference_weights` via the per-decision λ scalar. The γ formula is

$$\gamma_i = \tanh\!\left(\frac{\alpha_i}{|\beta_i|^{\lambda}} + |\beta_i|^{1-\lambda}\, g_m\right)$$

and is **mathematically valid for any real λ** as long as $|\beta_i| > 0$. The behavior at the tails changes:

- **λ > 0 (bearish):** $|\beta_i|^\lambda$ grows with $|\beta_i|$; high-beta names get α damped and market exposure scaled down → defensive tilt toward low-beta names.
- **λ < 0 (bullish):** $|\beta_i|^\lambda = |\beta_i|^{-|\lambda|}$ shrinks with $|\beta_i|$; α and market terms both amplified for high-beta names → risk-on tilt.

The 1e-8 floor on `RF = max(abs(βᵢ)^lambda, 1e-8)` still does the right thing: it only activates when the exponent drives `abs(βᵢ)^lambda → 0` (i.e., positive λ with small β). For negative λ the exponent flips and `abs(βᵢ)^lambda → ∞` for small β, which makes `α/RF → 0` harmlessly. The comment in `compute_preference_weights` was updated accordingly.

The `tanh` squash continues to bound γ ∈ (-1, 1), so γ output is safe even at extreme λ.

## Calibrating G

The default `G = 1.0` is a placeholder. Realistic EMA crossovers (short/long ratio) sit within ~±5% in calm markets, ±10–15% in trending markets. With `G = 1.0` that yields λ ∈ ~±0.05 to ±0.15, which is **much smaller** than the sigmoid form's effective range ([0, 1]) — the regime-lens effect will be muted unless G is scaled up.

Rough mapping (typical SPY EMA gap of ±5%):

| G    | Typical \|λ\| | Comparable to old sigmoid range? |
|------|---------------|-----------------------------------|
| 1    | 0.05          | very weak                         |
| 10   | 0.5           | comparable to mid sigmoid         |
| 20   | 1.0           | strong; |β|^λ varies meaningfully |
| 100  | 5.0           | extreme; risk of overflow in $|β|^{1-λ}$ for high-β names |

**Chosen value: G = 20.0 (2026-05-21, revised).** Initially G = 50 was wired in based on the G sweep alone, but the subsequent backtest re-run + β-bucket reanalysis (see sections below) revealed that G = 50 sat in the worst spot: strong enough to break the existing σ_max / w_max / K_turnover sweet-spots but not strong enough to deliver the bearish-side β-reversal (which requires G ≳ 100 plus the open-item defensive cap). G = 20 preserves clean bullish-side tilt (mean |Δγ| ≈ 0.02) without breaking the live engine's existing constraint calibration. Wired into all call sites:
- `code/src/Backtest.jl:181` — `compute_lambda(short, long; G = 20.0)`
- `scripts/02_train_bandit.jl:41` — `compute_lambda(short_ema, long_ema; G = 20.0)`
- `scripts/03_train_bandit_mc.jl:38` — `compute_lambda(short_ema, long_ema; G = 20.0)`
- `constrained-CD-with-MPC-paper-trade-trial/code/src/decide.jl` — `const _LAMBDA_GAIN = 20.0` threaded through the sole `compute_lambda` call site.

Companion fix at the same time: `Backtest.jl:183` fallback (when the EMA window is shorter than 21 days) was `λ_t = 0.5` — the sigmoid form's neutral midpoint. Under the signed form, neutral is 0.0; the old constant would have introduced a fixed mild bearish lean independent of market state. Updated to `λ_t = 0.0`.

## Tests

The previous test asserted `all(0.0 .<= λ .<= 1.0)` — that invariant is gone. Replaced (`code/test/test_sim.jl:53-74`) with three checks against the new sign convention:

1. Monotonically rising prices → `λ[end] < 0` (bullish).
2. Monotonically falling prices → `λ[end] > 0` (bearish).
3. Gain G scales the signal linearly: `λ(G=10) == 10 · λ(G=1)`.

`compute_preference_weights` test untouched (`lambda = 0.5` is still a valid signed value; tanh-bounded assertion still holds).

Full suite verified: **155/155 tests pass.**

## To Patch the Second Engine

1. Locate that engine's `compute_lambda` (or equivalent regime-lens function).
2. Replace the sigmoid body with the signed gain-based form above. Rename `θ → G` (or whatever keyword fits its style).
3. Audit consumers — anywhere the old λ was treated as a probability or bounded weight (e.g., used as a mixing coefficient, clipped to [0,1], indexed into a regime table) needs review. The signed form will break those assumptions.
4. Update any test that asserts `λ ∈ [0, 1]`.
5. Set G at the engine's primary `compute_lambda` call site. Match the value used here once chosen.
6. If both engines share log/dashboards, expect λ histograms to shift from $[0, 1]$ to a signed distribution centered near 0 — alerting/monitoring thresholds may need adjustment.

## Open Item (for follow-up, not blocking)

- Numerical edge case: in the γ formula, $|\beta_i|^{1 - \lambda} \cdot g_m$ can produce `Inf` if $|\beta_i| \to 0$ with large positive $1 - \lambda$ (i.e., large negative λ); combined with `g_m = 0` this gives `0 · Inf = NaN`. Realistic G keeps |λ| small enough that this does not fire, but if G is scaled aggressively (G ≳ 100) we should add a defensive cap on the exponentiated term. Track this if you push G high.

## G Sweep Results (2026-05-21)

Ran `scripts/lambda_g_sweep.jl` over G ∈ {0.5, 1, 5, 10, 20, 50, 100, 200} on the 2014–2024 SPY EMA series and the frozen 33-ticker basket. Findings:

| G    | λ min   | λ med  | λ max   | % bear | % γ≤0  | mean pref/K | % γ sign flip vs G=0 | mean \|Δγ vs G=0\| | NaN/Inf? |
|------|---------|--------|---------|--------|--------|-------------|----------------------|-------------------|----------|
| 0.5  | -0.024  | -0.006 |  0.049  | 21.0%  | 45.7%  | 17.9 / 33   | 0.0%                 | 0.0005            | no       |
| 1    | -0.047  | -0.012 |  0.098  | 21.0%  | 45.7%  | 17.9 / 33   | 0.0%                 | 0.0010            | no       |
| 5    | -0.237  | -0.061 |  0.491  | 21.0%  | 45.7%  | 17.9 / 33   | 0.0%                 | 0.0051            | no       |
| 10   | -0.474  | -0.123 |  0.982  | 21.0%  | 45.7%  | 17.9 / 33   | 0.0%                 | 0.0102            | no       |
| 20   | -0.949  | -0.245 |  1.964  | 21.0%  | 45.7%  | 17.9 / 33   | 0.0%                 | 0.0202            | no       |
| 50   | -2.371  | -0.613 |  4.909  | 21.0%  | 45.7%  | 17.9 / 33   | 0.0%                 | 0.0480            | no       |
| 100  | -4.743  | -1.226 |  9.818  | 21.0%  | 45.7%  | 17.9 / 33   | 0.0%                 | 0.0872            | no       |
| 200  | -9.485  | -2.451 | 19.637  | 21.0%  | 45.7%  | 17.9 / 33   | 0.0%                 | 0.1447            | no       |

### Reading the sweep

1. **Numerical stability — green.** No NaN or Inf anywhere across (day, ticker) pairs for G up to 200. The 1e-8 RF floor and the `tanh` squash together absorb every extreme combination encountered on real data. The conservative-cap follow-up flagged above is therefore not urgent in the realistic G range.

2. **λ-distribution constants under G scaling.** % bear (fraction of days with λ > 0) is invariant at 21.0% across all G — expected, since G is a positive multiplier and sign(λ) = sign(short_ema/long_ema − 1) is independent of G. λ-spread scales linearly.

3. **STRUCTURAL FINDING — the lens does not change *which* names are preferred.** % γ sign flip vs the G = 0 (lens-off) baseline is exactly 0.0% for every G tested. The mean number of preferred names per day stays at 17.9 / 33 and the global % γ ≤ 0 stays at 45.7%, independent of G.

   This is not a numerical coincidence; it is an algebraic property of the γ formula:

   $$\gamma_i = \tanh\!\left(\frac{\alpha_i}{|\beta_i|^{\lambda}} + |\beta_i|^{1-\lambda}\, g_m\right) = \tanh\!\left(|\beta_i|^{1-\lambda} \cdot \left(\frac{\alpha_i}{|\beta_i|} + g_m\right)\right).$$

   The prefactor $|\beta_i|^{1-\lambda} > 0$ for all real λ (since $|\beta_i| > 0$), so

   $$\mathrm{sign}(\gamma_i) = \mathrm{sign}\!\left(\frac{\alpha_i}{|\beta_i|} + g_m\right) = \mathrm{sign}(\alpha_i + |\beta_i| \cdot g_m),$$

   which is **independent of λ**. The regime-lens only rescales magnitudes via the $|\beta_i|^{1-\lambda}$ prefactor; it cannot move a name across the preferred / non-preferred boundary.

4. **Magnitude modulation does happen.** `mean |Δγ vs G=0|` grows from 0.0005 at G = 0.5 to 0.145 at G = 200 — a real, monotonic effect on individual γ values. Since the Cobb-Douglas allocator weights names by $\gamma_i / \sum_j \gamma_j$ inside the closed-form (or by relative magnitudes through the log objective in the constrained variant), magnitude shifts of this order *do* propagate to weights and trade deltas. The lens is therefore a weight-tilt knob, not a name-selection knob.

5. **`tanh` saturates early.** γ min/max already reach −1.000 / +1.000 at G = 0.5 — driven by high-|g_m| days. Past G ≈ 10, an increasing share of (day, ticker) pairs sit in tanh's saturating tails, which compresses the differential effect of further increases in G.

### β-Bucket × Regime Tilt (empirical, same sweep)

The sign-invariance result above said nothing about *which* preferred names attract weight. To see the actual regime tilt, split the basket into β-terciles (|β| ≤ 0.81 = low, |β| ≥ 1.07 = high) and split days by λ sign, then average the normalized allocator weight $w_i = \gamma_i / \sum_j \gamma_j^+$ over preferred (day, ticker) pairs:

| G   | Regime   | low-β mean w | mid-β mean w | high-β mean w | high / low ratio |
|-----|----------|--------------|--------------|---------------|------------------|
| 1   | bullish  | 0.0278       | 0.0350       | 0.0355        | 1.28             |
| 1   | bearish  | 0.0288       | 0.0334       | 0.0331        | 1.15 (no reversal) |
| 20  | bullish  | 0.0255       | 0.0355       | 0.0373        | 1.46             |
| 20  | bearish  | 0.0302       | 0.0330       | 0.0321        | 1.06 (effectively flat) |
| 100 | bullish  | 0.0179       | 0.0365       | **0.0441**    | **2.46**         |
| 100 | bearish  | **0.0342**   | 0.0319       | 0.0290        | 0.85 (**reversal**: low > high) |

The lens does precisely what the algebra predicts:

- **Bullish (λ < 0):** the exponent $1 - \lambda > 1$ amplifies high-β names. High-β weight rises and low-β weight falls. Visible at any G; pronounced at G ≥ 50.
- **Bearish (λ > 0):** the exponent $1 - \lambda < 1$, and only when **λ > 1** does the exponent go negative and start amplifying low-β names instead. With this EMA setup the median bearish λ is small (≈ 0.012 at G=1), so for the lens to push the bearish-day exponent past 1 you need G ≥ ~50. At G = 1 and G = 20 the bearish row shows essentially no reversal; at G = 100 it does (low-β weight 0.034 vs high-β 0.029).

This is the regime tilt mechanism. It is **magnitude-based, not selection-based** — names don't cross the preferred / non-preferred boundary, but their share of the preferred-set weight mass shifts systematically with β as λ changes.

### Implications for G calibration

- **Safe range:** anywhere in [0.5, 200] is numerically fine.
- **Useful range for the bullish-tilt only:** G ∈ [10, 50] amplifies high-β in rising markets. Bearish-side tilt is too weak here to reverse the ordering.
- **Useful range for both directions:** **G ≈ 50–100**. At G = 100 the bullish high/low ratio reaches 2.5× and the bearish row reverses to a low > high ordering. This is the band where the lens actually swings the basket between offensive and defensive postures.
- **Diminishing returns past G ≈ 100–200** because of tanh saturation on high-|g_m| days.
- Recommendation if the running engine needs a single G: **start at G = 75**, monitor mean β-exposure by regime, and tune up or down based on how aggressive a swing you want.

### Implications for downstream design

The regime-lens operates on the **continuous γ-magnitude axis**, not the discrete preferred / non-preferred axis. That has three concrete downstream consequences:

- **The lens cannot trigger the cash regime by itself.** The `:no_preferred` fallback in `code/src/Backtest.jl` and `code/src/Allocator.jl` fires when every γ_i ≤ 0. Per the sign-invariance derivation, that condition is fully determined by α and g_m, never by λ. Cash regimes track broad market drops (via g_m and the α distribution), not the lens setting.
- **The lens *does* shift composition inside the preferred set.** The $|\beta_i|^{1-\lambda}$ prefactor reweights preferred names by β in a regime-dependent way (see the β-bucket table above). This is the channel the lens uses to swing the basket between offensive (bullish, high-β-tilted) and defensive (bearish, low-β-tilted) postures.
- **Dashboards / alerts that key off "number of preferred names" or "% in cash" will not respond to G changes.** Those signals see the lens-invariant axis. To monitor the lens's effect you need a composition-aware metric — β-weighted portfolio exposure, the high-vs-low-β weight ratio, or an entropy / concentration measure over the preferred set. Add one of these to the operator dashboard before turning G up.

## Backtest Re-Run Results (2026-05-21, G = 50)

Re-ran `scripts/06_backtest_mc.jl` and the four sweep scripts (`07–10`) against the frozen 33-ticker basket. The bandit / basket pipeline (`02–04`) was left untouched because the live test is committed to the current basket. Before re-running, the previous artifacts were copied to `*.pre_lambda_swap.jld2` for delta comparison; `scripts/lambda_swap_deltas.jl` produces the report below.

### Headline bake-off (median over 20 seeds)

| Strategy                  | Sharpe new / old / Δ      | MaxDD% new / old / Δ      | W_T/W_0 new / old / Δ |
|---------------------------|---------------------------|---------------------------|------------------------|
| CDWithMPCStrategy         | +1.350 / +2.079 / **−0.729** | 9.9 / 8.4 / +1.5         | 1.272 / 1.398 / −0.126 |
| ConstrainedCDWithMPC      | +1.353 / +1.681 / **−0.327** | 9.2 / 7.5 / +1.8         | 1.278 / 1.266 / +0.012 |
| UnconstrainedCDStrategy   | −1.257 / −1.263 / +0.005     | 31.2 / 24.7 / **+6.5**   | 0.699 / 0.772 / −0.073 |
| CostAwareMVStrategy       | +0.715 / +0.732 / −0.017     | 9.4 / 9.7 / −0.2         | 1.123 / 1.126 / −0.003 |
| MinVarBuyHoldStrategy     | +1.199 / +1.199 / 0.000 ✓    | 11.0 / 11.0 / 0.0 ✓      | 1.229 / 1.229 / 0.000 ✓ |
| EqualWeightStrategy       | +1.160 / +1.160 / 0.000 ✓    | 15.9 / 15.9 / 0.0 ✓      | 1.247 / 1.247 / 0.000 ✓ |

MinVar and EqualWeight rows are unchanged to floating-point precision — confirms the test is isolating the λ effect cleanly (both strategies are λ-independent). The MPC strategies regressed: CDWithMPC took the largest hit because it runs the closed-form CD with no covariance / turnover / concentration guards, so the stronger β-tilted γ propagates straight to weights. The constrained variant absorbs half the shock because σ_max + K_turnover + w_max clip the tilt. UnconstrainedCD was already broken; the new λ deepens its drawdown by 6.5 points.

### Sweep sweet-spots — all four moved toward *looser* constraints

| Sweep        | Old optimum    | New optimum         | New Sharpe at old optimum | Direction      |
|--------------|----------------|---------------------|----------------------------|----------------|
| σ_max        | 1.5 (tight)    | ≥ 3 (effectively off) | 0.889 (was 1.977)         | **loosen**     |
| w_max        | 0.7 (off)      | 0.15 (tight cap)    | 1.381 (was 2.079)         | **tighten**    |
| K_turnover   | 25 (tight)     | ≥ 100 (loose)       | 0.901 (was 1.751)         | **loosen**     |
| cash_revisit | 5              | 5 (unchanged)       | 2.009 (was 2.302)         | unchanged level |

The σ_max and K_turnover moves go the same direction — under G = 50, the regime-lens produces stronger weight tilts and the engine wants more headroom to take and to churn the implied position. The w_max move goes the *opposite* way: now that the lens concentrates weight on a few β-favored names, the previously-inactive w_max = 0.7 cap doesn't bite and a tight 0.15 cap is needed to prevent a single name from running too far. Even at every new sweet spot the absolute Sharpe is below the pre-swap peak — the lens is doing more, but the system isn't extracting better risk-adjusted return from it at G = 50.

### What this means for the live test

If the running engine inherited the headline-bake-off defaults (σ_max ≈ 2.5, w_max ≈ 0.7, K_turnover ≈ 25), it is now running with **two constraints near or past their old sweet spots** while the lens applies a noticeably stronger β-tilt. Expected behavior in the live window:

- More frequent trade churn from the K_turnover = 25 cap binding under stronger day-over-day weight swings.
- Suppressed responsiveness when σ_max ≈ 2.5 would let the engine take a position — the new optimum is σ_max ≥ 3.
- Concentration risk on β-favored names, since w_max ≈ 0.7 is effectively off and the new lens wants to concentrate.

Three directions to choose from, **in order of disruption to the live test**:

1. **Accept the hit** — keep G = 50 and the current constraint defaults. Defensible only if the signed λ is required for live correctness regardless of offline Sharpe.
2. **Reduce G** to ≈ 20. The β-bucket sweep showed G = 20 still gives clean bullish tilt with much smaller weight perturbations. Smaller disruption to existing constraint calibrations; abandons the bearish-side reversal.
3. **Keep G = 50 and re-tune constraints** to the new sweet spots: σ_max ≥ 3, w_max = 0.15, K_turnover ≥ 100, cash_revisit = 5. Largest config delta to deploy; offline-optimal.

The right choice depends on what the live test is measuring. If it's measuring the signed-λ behavior itself, option (1) preserves the experiment. If it's measuring expected portfolio performance, option (3) is what offline now argues for. Option (2) splits the difference.

### Artifacts produced

- `scripts/lambda_swap_deltas.jl` — comparison report generator. Loads each `*.jld2` and the corresponding `*.<baseline>.jld2` backup, prints the tables above. Takes an optional baseline suffix as ARGS[1] (default `pre_lambda_swap`); pass `g50` to diff against the G=50 snapshot instead.
- `scripts/data/*.pre_lambda_swap.jld2` (local only; gitignored) — pre-swap backups (original sigmoid). Keep until the G-value decision is finalized.
- `scripts/data/*.g50.jld2` (local only; gitignored) — G=50 snapshot taken before the G=20 re-run.

## Backtest Re-Run Results (2026-05-21, G = 20 revision)

After the G=50 results showed material offline regression, G was reduced to 20 in the hope of preserving the bullish-side tilt while minimizing disruption to the existing constraint calibrations. The re-run shows that hypothesis was wrong: **G = 20 is roughly comparable to G = 50 in absolute Sharpe and slightly worse on the constrained variant**, and neither recovers anywhere close to the pre-swap baseline.

### Headline bake-off (median over 20 seeds)

| Strategy                  | pre-swap | G = 50  | G = 20  | G=20 vs pre Δ | G=20 vs G=50 Δ |
|---------------------------|----------|---------|---------|---------------|----------------|
| CDWithMPCStrategy         | +2.079   | +1.350  | +1.339  | **−0.740**    | −0.011 (DD +7.7 pts) |
| ConstrainedCDWithMPC      | +1.681   | +1.353  | **+1.188** | **−0.492**    | **−0.165**     |
| UnconstrainedCDStrategy   | −1.263   | −1.257  | −1.307  | −0.045        | −0.050         |
| CostAwareMVStrategy       | +0.732   | +0.715  | +0.716  | −0.016        | +0.001         |
| MinVarBuyHoldStrategy     | +1.199   | +1.199  | +1.199  | 0.000 ✓       | 0.000 ✓        |
| EqualWeightStrategy       | +1.160   | +1.160  | +1.160  | 0.000 ✓       | 0.000 ✓        |

### Why G = 20 underperformed expectations

The β-bucket sweep had already shown that at G = 20 the bearish-side reversal does not materialize — bearish-day mean weights stay at 0.030 / 0.033 / 0.032 across low / mid / high-β buckets, essentially flat. G = 20 keeps the **offensive bullish tilt** (high-β amplified when λ < 0) but provides **no defensive rotation** in bearish regimes. G = 50 had *partial* bearish reversal which gave some defensive cover. So lowering G from 50 to 20 removed the defensive mechanism while leaving the offensive one in place — strictly worse than either keeping G = 50 or pushing past G = 100 for full reversal.

The CDWithMPC drawdown nearly doubled (9.9% → 17.6%) going from G = 50 to G = 20, consistent with the loss of defensive rotation in falling-market windows.

### Sweep sweet-spots — almost identical to G = 50

| Sweep        | pre-swap | G = 50         | G = 20         | Direction       |
|--------------|----------|----------------|----------------|-----------------|
| σ_max        | 1.5      | ≥ 3            | 2.5            | slight tighten vs G=50 |
| w_max        | 0.7      | 0.15           | 0.15           | unchanged       |
| K_turnover   | 25       | ≥ 100          | ≥ 100          | unchanged       |
| cash_revisit | 5        | 5              | 5              | unchanged       |

The constraint sweet spots are essentially **G-invariant** in the [20, 50] range. σ_max moves slightly tighter at G = 20 (back toward 2.5), but the absolute Sharpe at every grid point is within ~0.04 of the G = 50 number. **The G dial does not move the constraint calibration; it only moves the absolute Sharpe by a small amount, dominated by the bearish-side regime tilt being on (G = 50) vs off (G = 20).**

### Revised takeaway

Three things I was wrong about, captured here so we don't repeat them:

1. **"G = 20 is the lower-disruption choice."** False. Lower G doesn't take pressure off the constraints (sweet spots are nearly identical); it just removes the partial defensive rotation. Net effect on the headline strategy is *slightly negative* vs G = 50, not neutral.
2. **"More guardrails → smaller hit."** Half right. The constrained variant absorbs more shock than the closed-form CDWithMPC, but the absorption is partial — the constrained strategy still loses 0.33 (G=50) or 0.49 (G=20) Sharpe vs pre-swap. The constraints are not a free hedge against the λ change.
3. **"G ≈ 50 should give meaningful bearish reversal."** Only partial. Looking at the G sweep, full reversal needs G ≳ 100. G = 50 sits in an awkward middle: enough to break old constraint calibrations but not enough for the bearish defensive rotation it was meant to enable.

### Where this leaves the live test

- **G choice in [20, 50] is offline-roughly-equivalent**, ~0.5 Sharpe below pre-swap regardless. G = 50 marginally better on the constrained strategy.
- **Constraint re-tuning is the real lever.** σ_max ≥ 3, w_max = 0.15, K_turnover ≥ 100 — same recipe at both G values. If the live test allows a constraint update, that's where the offline-optimal Sharpe is.
- **Full bearish reversal needs G ≳ 100** AND the defensive cap on the open-item γ-overflow edge case. That's a second-order project, not a quick swap.
- **If signed λ is required for live correctness regardless of offline Sharpe** (the original reason this swap happened), pick G to match the live-engine intent; offline backtest cannot adjudicate between G = 20 and G = 50 by Sharpe alone.
