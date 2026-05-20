# Theory

The constrained Cobb-Douglas + MPC framework is a three-layer decision system. A **universe layer** picks the basket once, before the backtest begins. An **allocation layer** solves a small convex program for target weights given a basket and the current state. An **MPC layer** decides *when* to re-allocate by comparing the realized portfolio path against a forward-projected confidence band. This page restates the math behind each layer and points to the function that implements it.

## 1. Constrained Cobb-Douglas allocation

The allocator maximizes a log-utility Cobb-Douglas objective over a budgeted, risk-budgeted, turnover-budgeted, concentration-capped, long-only basket of ``K`` assets:

```math
\max_{n_i \,\ge\, 0}\; \sum_{i=1}^{K} \gamma_i \, \log\!\left(n_i\right)
```

subject to

```math
\begin{aligned}
\sum_{i=1}^{K} n_i \, p_i \;&=\; B
   && \text{(budget identity)} \\
w^{\top} \Sigma\, w \;&\le\; \sigma_{\max}^{2}
   && \text{(covariance budget)} \\
\bigl\lVert n - n_{\text{prev}} \bigr\rVert_{1} \cdot \bar{c} \;&\le\; K_{\text{turnover}}
   && \text{(turnover budget)} \\
w_i \;&\le\; w_{\max}
   && \text{(concentration cap per name)} \\
n_i \;&\ge\; 0
   && \text{(long-only)}
\end{aligned}
```

where ``w_i = n_i \, p_i / B`` is the dollar weight of asset ``i``, ``\Sigma`` is the single-index-model (SIM) implied covariance defined below, ``n_{\text{prev}}`` is the position vector from the prior MPC decision, ``\bar{c}`` is the average per-share transaction cost, and ``K_{\text{turnover}}`` is a dollar turnover budget per decision. The objective is concave in ``n`` when ``\gamma_i > 0``, and the feasible region is the intersection of a budget hyperplane, a single convex-quadratic ellipsoid (the covariance budget), and box-and-simplex-style linear constraints. For ``K`` of order 20-50 the problem solves in milliseconds via JuMP with Clarabel or SCS. Implemented in [`solve_constrained_cd`](api/allocator.md).

### 1.1 The preference vector γ

The preference exponent for asset ``i`` is built from its SIM regression coefficients and the prevailing market regime, with no news term:

```math
\gamma_i \;=\; \tanh\!\left(\, \frac{\alpha_i}{\,\lvert \beta_i \rvert^{\,\lambda}\,} \;+\; \lvert \beta_i \rvert^{\,1-\lambda} \cdot g_{m}\, \right)
```

``\alpha_i, \beta_i, \sigma_{\varepsilon,i}`` are estimated by the SIM regression on daily SPY-relative log returns. ``\lambda \in [0, 1]`` is the regime-lens parameter from an EMA crossover on the market index — at ``\lambda \to 0`` the preference is driven by name-specific alpha, at ``\lambda \to 1`` it is driven by market exposure. ``g_m`` is the smoothed market growth rate. The ``\tanh`` squashes the score into ``(-1, 1)`` so that ``\gamma_i`` is bounded and the log-utility objective is well posed. Build a ``\gamma`` vector with [`compute_preference_weights`](api/sim.md).

### 1.2 The SIM-implied covariance matrix Σ

The covariance matrix is decomposed into a one-factor structure:

```math
\Sigma \;=\; \sigma_{m}^{2}\, \beta\, \beta^{\top} \;+\; \mathrm{diag}\!\left(\sigma_{\varepsilon,1}^{2},\, \ldots,\, \sigma_{\varepsilon,K}^{2}\right)
```

``\sigma_{m}^{2}`` is the annualized variance of the market index and ``\beta = (\beta_{1}, \ldots, \beta_{K})^{\top}`` collects the per-asset market loadings from the SIM. No separate sample-covariance estimate is needed; ``\Sigma`` is a function of the same parameters that drive ``\gamma``. Build it with [`build_sim_covariance`](api/sim.md).

### 1.3 The two scalar parameters that drive behavior

Two interpretable scalars set the operating point of the constrained CD allocator:

- ``\sigma_{\max}`` — the annualized portfolio volatility cap. Encodes client risk tolerance; calibrated by client conversation, not by backtest. The sensitivity of the strategy to ``\sigma_{\max}`` is mapped by [`scripts/07_sigma_max_sweep.jl`](https://github.com/varnerlab/modified_cobb_douglas_example/blob/main/scripts/07_sigma_max_sweep.jl).
- ``K_{\text{turnover}}`` — the per-decision dollar turnover budget. Controls how much the new allocation may differ from the prior one and is the lever that prevented the live engine's micro-churn failure mode. Mapped by [`scripts/09_k_turnover_sweep.jl`](https://github.com/varnerlab/modified_cobb_douglas_example/blob/main/scripts/09_k_turnover_sweep.jl).

Both have natural units that the strategist can defend in a client conversation; neither is a ``\kappa``-style risk-aversion coefficient with no real-world meaning. A third lever, the per-name concentration cap ``w_{\max}``, is mapped by [`scripts/08_w_max_sweep.jl`](https://github.com/varnerlab/modified_cobb_douglas_example/blob/main/scripts/08_w_max_sweep.jl).

## 2. Model-predictive control

### 2.1 Forward projection and the in-spec band

At MPC decision time ``t``, the allocator forward-projects portfolio value ``V_{\tau}`` for ``\tau = t+1, t+2, \ldots, t+T`` and records the cross-path mean ``\mu_{\tau}`` and standard deviation ``\sigma_{\tau}`` at each horizon. The current allocation is **in spec** at ``\tau`` if and only if

```math
\mu_{\tau} \;-\; z \cdot \sigma_{\tau} \;\le\; V_{\tau} \;\le\; \mu_{\tau} \;+\; z \cdot \sigma_{\tau}
```

with defaults ``z = 1.96`` (95% confidence band) and ``T = 21`` trading days (one-month horizon). Both are tunable.

The forward projection itself is a hybrid SPY-JumpHMM + SIM Monte Carlo. For each of ``N`` paths indexed by ``j`` and each forward step ``\tau``:

1. Draw an SPY log return ``g_{m,\tau}^{(j)}`` from a JumpHMM marginal calibrated on the training window.
2. Generate per-asset log returns conditional on the market path:

   ```math
   g_{i,\tau}^{(j)} \;=\; \alpha_{i} \;+\; \beta_{i}\cdot g_{m,\tau}^{(j)} \;+\; \varepsilon_{i,\tau}^{(j)}, \qquad \varepsilon_{i,\tau}^{(j)} \,\sim\, \mathcal{N}\!\left(0,\, \sigma_{\varepsilon,i}^{2}\right)
   ```

   Cross-asset coupling is carried entirely by ``\beta`` through the shared market factor; the idiosyncratic terms are independent across assets and across time.

3. Compound to portfolio value under the current weights ``w``:

   ```math
   V_{\tau}^{(j)} \;=\; V_{0} \cdot \prod_{s=1}^{\tau}\!\left( 1 + \sum_{i=1}^{K} w_{i}\cdot\!\left(e^{\,g_{i,s}^{(j)}} - 1\right)\right)
   ```

   Wealth dynamics are continuously compounded within each day and discretely compounded across days.

4. Aggregate the band statistics: ``\mu_{\tau} = \operatorname{mean}_{j} V_{\tau}^{(j)}`` and ``\sigma_{\tau} = \operatorname{std}_{j} V_{\tau}^{(j)}``.

A lognormal closed-form approximation using only the SIM moments ``\mu_{p} = w^{\top}\!\alpha`` and ``\sigma_{p}^{2} = w^{\top}\Sigma\, w`` is computed in parallel as a validation path; large divergences from the Monte Carlo band are logged for inspection. The trigger decision uses the JumpHMM Monte Carlo band. Implemented in [`forward_project`](api/mpc.md) and [`forward_project_closed_form`](api/mpc.md).

### 2.2 Trigger logic

Let ``\tau = t_{\text{now}} - t_{\text{last decision}}`` be the elapsed time since the most recent allocation. The MPC layer re-fires the allocator when **any** of the following conditions holds, evaluated in order:

1. **Cash revisit** — the prior decision returned the defensive ``\varepsilon``-pin (all cash) regime *and* ``\tau \ge \texttt{cash\_revisit\_interval}``. This fast-paths re-evaluation after a defensive flight to cash so the strategy can re-enter once conditions improve. Mapped by [`scripts/10_cash_revisit_sweep.jl`](https://github.com/varnerlab/modified_cobb_douglas_example/blob/main/scripts/10_cash_revisit_sweep.jl).
2. **Drawdown circuit-breaker** — realized drawdown from the wealth peak exceeds ``D_{\max}``.
3. **Horizon elapsed** — ``\tau \ge T``; the projection horizon has been reached, so the band is no longer informative.
4. **Band exit** — ``V_{t}`` lies outside ``[\mu_{\tau} - z\,\sigma_{\tau},\; \mu_{\tau} + z\,\sigma_{\tau}]`` on the realized path.

Between triggers, the engine submits no orders. This is the discipline that fixes the live failure mode: cadence is event-driven by realized state, not by a fixed clock. Implemented in [`check_trigger`](api/mpc.md).

## 3. Per-sector bandit for universe selection

The universe layer picks the ``K``-name basket once, before backtest. Eleven parallel ``\varepsilon``-greedy bandits — one per GICS sector — each pick ``q_{s}`` tickers from their sector's candidate set, with sector quotas summing to the basket size:

```math
\sum_{s=1}^{11} q_{s} \;=\; K_{\text{basket}}
```

Quotas are assigned proportional to sector market capitalization. Implemented in [`assign_quotas`](api/bandit.md) and [`train_sector_bandit`](api/bandit.md).

### 3.1 Reward signal

The reward for picking sector-``s`` sub-basket ``\mathcal{B}`` at decision day ``d`` is the **sector-relative** 21-day forward log return of the Cobb-Douglas-allocated sub-basket:

```math
R_{s}(\mathcal{B}, d) \;=\; r_{\mathcal{B}}\!\left(d,\, d+21\right) \;-\; r_{\mathrm{EW},\,s}\!\left(d,\, d+21\right)
```

``r_{\mathrm{EW},\,s}`` is the equal-weight return inside sector ``s`` over the same window. Subtracting it strips out the within-sector market factor that the bandit did not choose; what remains is the basket-selection signal. Computed by [`sector_relative_reward`](api/bandit.md); the CD-on-sub-basket return that defines ``r_{\mathcal{B}}`` is computed by [`cd_basket_return`](api/bandit.md).

### 3.2 Training and freezing

The bandits train on roughly a decade of S&P 500 daily closes (a 2014-01-03 to 2024-12-31 window in the shipped artifacts) restricted to tickers with full OHLC coverage on every training and hold-out day. Multiple seeds are trained; the median-Sharpe seed is selected as the canonical basket and frozen for the entire hold-out window. The learning adds incremental edge over the sector quotas alone — random-per-sector selection with the same quotas does not match the trained bandit's hold-out Sharpe.

## 4. References

- Cobb, C. W. and Douglas, P. H. (1928). "A Theory of Production." *American Economic Review* — the original Cobb-Douglas utility.
- Sharpe, W. F. (1963). "A Simplified Model for Portfolio Analysis." *Management Science* — the single-index model that produces ``(\alpha, \beta, \sigma_{\varepsilon})`` and the SIM-implied ``\Sigma``.
- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance* — the mean-variance baseline that strategy 4 of the backtest harness reproduces.
- Camacho, E. F. and Bordons, C. *Model Predictive Control*. Springer — classical MPC: horizon, projection, recourse.
- Boyd, S. *et al.* (2017). "Multi-Period Trading via Convex Optimization." *Foundations and Trends in Optimization* — the closest analogue in finance to the discipline described here.
- Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem." *Machine Learning* — analysis of the ``\varepsilon``-greedy bandit.
