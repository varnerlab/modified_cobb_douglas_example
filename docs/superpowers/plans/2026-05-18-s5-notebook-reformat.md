# S5 Notebook Reformat — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reformat `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb` into the four-section bandit-notebook layout (Introduction / Theory Recap / Results / Summary) and re-run the pipeline at `K_basket = 33` (uniform 3-per-sector quotas) so the notebook displays the new basket size and statistically anchored single-seed visualizations.

**Architecture:** Two phases. **Phase A** edits one constant in `scripts/03_train_bandit_mc.jl` and re-runs `03 → 04 → 06` (skipping `01`, `02`, `05`). **Phase B** edits the notebook cell-by-cell via NotebookEdit: replaces the Introduction and Theory Recap cells, inserts a Section 2 opener with a Data Windows blockquote and a Frozen Basket roster, demotes the four Results subsection headers under one `## Section 2: Results`, retargets the bake-off / wealth-curves / trigger-reasons cells to read from `bt_mc["per_seed_results"]` at a canonical median-Sharpe seed (eliminating the dropped `backtest_results.jld2`), and replaces the top-level Disclaimer with a `## Summary` cell containing a headline paragraph, Key Takeaways blockquote, and `### Disclaimer`.

**Tech Stack:** Julia (1.10+), JuMP + Clarabel for constrained CD, JLD2 for artifact persistence, NotebookEdit tool for cell-by-cell notebook edits, Jupyter for end-to-end notebook execution.

**Spec:** `docs/superpowers/specs/2026-05-18-s5-notebook-reformat-design.md`.

---

## Phase A — Pipeline re-run at K_basket = 33

### Task 1: Edit `K_BASKET = 33` in `03_train_bandit_mc.jl`

**Files:**
- Modify: `scripts/03_train_bandit_mc.jl:14`

- [ ] **Step 1: Apply the constant edit**

Use the Edit tool:
- `old_string`: `const K_BASKET = 22`
- `new_string`: `const K_BASKET = 33`

- [ ] **Step 2: Verify the edit**

Run: `grep -n 'K_BASKET' scripts/03_train_bandit_mc.jl`
Expected: `14:const K_BASKET = 33` (the other matches are `K_BASKET` references inside the script body and the `"K_BASKET" => K_BASKET` Dict entry).

- [ ] **Step 3: Commit the config change alone**

```bash
git add scripts/03_train_bandit_mc.jl
git commit -m "$(cat <<'EOF'
scripts/03: K_BASKET 22 → 33 for uniform 3-per-sector quotas

11 GICS sectors × 3 = 33; assign_quotas distributes K÷S per sector
with remainder = 0 here, so q_s = 3 uniformly. The notebook reformat
spec (docs/superpowers/specs/2026-05-18-s5-notebook-reformat-design.md)
calls for K=33; this is the only pipeline edit needed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Re-run script 03 (30-seed bandit MC at K=33)

**Files:**
- Read input: `scripts/data/sim_calibration.jld2` (existing, K-invariant)
- Write output: `scripts/data/per_sector_bandit_mc_results.jld2` (overwritten)

- [ ] **Step 1: Run script 03 in the background**

Use Bash with `run_in_background = true`:

```bash
cd /Users/jdv27/Desktop/julia_work/modified_cobb_douglas_example && julia --project=code scripts/03_train_bandit_mc.jl
```

This trains 11 per-sector bandits for each of 30 seeds (1001:1030). Wait for completion.

- [ ] **Step 2: Verify the artifact was written**

Run: `julia --project=code -e 'using JLD2; d = JLD2.load("scripts/data/per_sector_bandit_mc_results.jld2"); println("K_BASKET = ", d["config"]["K_BASKET"]); println("seeds = ", length(d["config"]["BANDIT_MC_SEEDS"])); println("quotas = ", d["quotas"])'`

Expected output:
```
K_BASKET = 33
seeds = 30
quotas = Dict("Industrials" => 3, "Health Care" => 3, "Information Technology" => 3, ...)  # all 11 sectors mapped to 3
```

If any quota ≠ 3 or seed count ≠ 30, STOP and investigate before proceeding.

---

### Task 3: Re-run script 04 (basket selection)

**Files:**
- Read input: `scripts/data/per_sector_bandit_mc_results.jld2` (from Task 2)
- Write output: `scripts/data/frozen_basket.jld2` (overwritten with 33 tickers)

- [ ] **Step 1: Run script 04**

```bash
cd /Users/jdv27/Desktop/julia_work/modified_cobb_douglas_example && julia --project=code scripts/04_select_basket.jl
```

This is fast — picks the median-score seed from the 30 MC seeds and writes the frozen basket.

- [ ] **Step 2: Verify the frozen basket shape (Phase A acceptance §3.4)**

Run: `julia --project=code -e 'using JLD2; b = JLD2.load("scripts/data/frozen_basket.jld2"); @assert length(b["tickers"]) == 33; @assert all(v == 3 for v in values(b["sector_quotas"])); @assert length(b["sector_quotas"]) == 11; println("OK: 33 tickers, 11 sectors × 3, seed = ", b["seed_id"])'`

Expected: `OK: 33 tickers, 11 sectors × 3, seed = <some integer in 1001:1030>` and exit 0.

If any assertion fails, STOP and investigate.

---

### Task 4: Re-run script 06 (20-seed backtest MC)

**Files:**
- Read input: `scripts/data/sim_calibration.jld2`, `scripts/data/frozen_basket.jld2`
- Write output: `scripts/data/backtest_mc_results.jld2` (overwritten)

- [ ] **Step 1: Run script 06 in the background**

```bash
cd /Users/jdv27/Desktop/julia_work/modified_cobb_douglas_example && julia --project=code scripts/06_backtest_mc.jl
```

This is the longest script: 20 seeds × 6 strategies × 326 hold-out days, with the two MPC strategies doing a 1000-path forward-projection MC at each trigger check. Run in background; expect several minutes.

- [ ] **Step 2: Verify the backtest MC artifact shape (Phase A acceptance §3.4)**

Run:
```bash
julia --project=code -e '
using JLD2, Statistics
bt = JLD2.load("scripts/data/backtest_mc_results.jld2")
strats = ["EqualWeightStrategy","MinVarBuyHoldStrategy","UnconstrainedCDStrategy",
          "CostAwareMVStrategy","CDWithMPCStrategy","ConstrainedCDWithMPCStrategy"]
@assert all(haskey(bt["summary"], s) for s in strats)
@assert length(bt["per_seed_results"]) == 20
@assert all(haskey(bt["per_seed_results"][i], s) for i in 1:20, s in strats)
sharpes = bt["summary"]["ConstrainedCDWithMPCStrategy"]["sharpe_mc"]
println("OK: 6 strategies × 20 seeds present")
println("Hold-out: ", bt["config"]["hold_out_start"], " to ", bt["config"]["hold_out_end"])
println("ConstrainedCDWithMPC Sharpe — min: ", round(minimum(sharpes); digits=3),
        ", median: ", round(median(sharpes); digits=3),
        ", max: ", round(maximum(sharpes); digits=3))
'
```

Expected: prints `OK: 6 strategies × 20 seeds present`, hold-out window, and Sharpe distribution stats.

If any assertion fails, STOP.

---

### Task 5: Decide whether to commit Phase A artifacts

**Files:**
- Possibly: `scripts/data/per_sector_bandit_mc_results.jld2`, `scripts/data/frozen_basket.jld2`, `scripts/data/backtest_mc_results.jld2`

- [ ] **Step 1: Check whether `.jld2` artifacts are tracked**

Run: `git status --short scripts/data/`
Also: `cat .gitignore | grep -E '\.jld2|scripts/data'`

If the `.jld2` files appear in `git status` as modified/untracked, they're tracked and should be committed. If they don't appear, they're gitignored — skip the commit and note it.

- [ ] **Step 2: If tracked, commit the artifacts**

```bash
git add scripts/data/per_sector_bandit_mc_results.jld2 scripts/data/frozen_basket.jld2 scripts/data/backtest_mc_results.jld2
git commit -m "$(cat <<'EOF'
Re-run pipeline at K_basket=33: bandit MC, basket selection, backtest MC

Phase A artifacts produced by 03 → 04 → 06 at K_BASKET=33 with uniform
3-per-sector GICS quotas. The notebook reformat (Phase B) reads these
artifacts. Script 05 was skipped because 06 already saves per-strategy
MyBacktestResult for all 20 seeds — see spec §3.2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If not tracked, skip this step entirely and proceed to Phase B. The artifacts on disk remain as the Phase B inputs.

---

## Phase B — Notebook reformat

All cell edits use the `NotebookEdit` tool with `notebook_path = "/Users/jdv27/Desktop/julia_work/modified_cobb_douglas_example/eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb"`.

To find a target cell's `cell_id`, run: `jq '.cells | to_entries | map({idx: .key, id: .value.id, type: .value.cell_type, head: (.value.source | join("") | .[0:80])})' <notebook>` before each task.

### Task 6: Replace the Introduction cell (cell 0)

**Files:**
- Modify: `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb` (cell 0, markdown)

- [ ] **Step 1: Locate the cell's ID**

```bash
jq -r '.cells[0] | {id, cell_type, head: (.source | join("") | .[0:80])}' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: a markdown cell whose head begins with `# Constrained Cobb-Douglas with MPC`. Note the `id` value.

- [ ] **Step 2: Replace cell 0 with the new Introduction**

Use NotebookEdit with `edit_mode = "replace"`, `cell_id` from Step 1, `cell_type = "markdown"`, and `new_source`:

```markdown
# Constrained Cobb-Douglas with MPC — Theory and Hold-Out Results

The live intraday Cobb-Douglas (CD) engine that ran from 2026-05-05 to 2026-05-15 on a paper account was discontinued at market close on 2026-05-18 with a flat-with-bleed P&L. The diagnostic was clean: the allocator solved unconstrained CD at a 30-min clock, the turnover gate was disabled, and γ jitter at 30-min cadence was being rounded into ±1-share orders on 5-10 tickers per fire. The engine had no notion of cost, no notion of risk, no notion of holding, and fired on a clock with no information content.

This notebook walks through the design that replaces it: a **constrained Cobb-Douglas allocator** (covariance + turnover + concentration budgets) wrapped in **model-predictive control (MPC)** discipline (forward-project, fire only when the realized path leaves a confidence band). The universe is a 33-ticker basket frozen from the S4 per-sector bandit (uniform 3-per-sector GICS quotas). We benchmark the design against five baselines on the 2025-2026 hold-out window with after-cost, after-tax accounting.

> **Learning Objectives:**
>
> - Formulate the constrained Cobb-Douglas allocation problem (budget + covariance + turnover + concentration) and recognize it as a convex program solvable in milliseconds at K=33.
> - Read the MPC discipline (forward projection band + trigger conditions) as a discipline that converts a clock-driven rebalancer into an event-driven one.
> - Compare the 6 strategies head-to-head on the 2025-2026 hold-out and interpret what each pairwise difference isolates (constraint effect vs trigger effect).

Let's walk through the theory and read the bake-off.
```

- [ ] **Step 3: Verify the cell renders**

Re-run the jq command from Step 1; confirm `head` now begins with `# Constrained Cobb-Douglas with MPC — Theory and Hold-Out Results\n\nThe live intraday`.

---

### Task 7: Replace the Theory Recap cell (cell 1)

**Files:**
- Modify: `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb` (cell 1, markdown)

- [ ] **Step 1: Locate the cell's ID**

```bash
jq -r '.cells[1] | {id, cell_type, head: (.source | join("") | .[0:80])}' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: markdown cell starting with `## Section 1: Theory Recap`. Note `id`.

- [ ] **Step 2: Replace with full Theory Recap content**

Use NotebookEdit `edit_mode = "replace"`, `cell_type = "markdown"`, `new_source`:

```markdown
___
## Section 1: Theory Recap

The decision system has three layers — universe (picked once by the S4 bandit), allocation (the constrained CD solver), and MPC (the forward-projection-band trigger that decides _when_ to re-allocate). Between MPC re-triggers the engine submits no orders. This section pins the math and the algorithm; the implementation lives in `code/src/Allocator.jl` and `code/src/Backtest.jl`.

> **Constrained CD optimization:**
>
> Given a basket of $K = 33$ tickers, prices $p \in \mathbb{R}^K_{>0}$, budget $B$, prior shares $n_{\text{prev}}$, SIM-implied covariance $\Sigma$, and per-share average cost $\bar{c}$, solve:
>
> $$\max_{n_i \ge 0} \; \sum_{i=1}^{K} \gamma_i \log(n_i)$$
>
> subject to
>
> $$\sum_i n_i p_i \le B, \quad w^\top \Sigma w \le \sigma_{\max}^2, \quad \bar{c}\,\|n - n_{\text{prev}}\|_1 \le K_{\text{turnover}}, \quad w_i \le w_{\max}, \quad n_i \ge 0$$
>
> where $w_i = n_i p_i / B$. The two knobs $\sigma_{\max}$ (annualized portfolio vol cap; default in growth-rate-vol units) and $K_{\text{turnover}}$ (dollar turnover budget per decision; default 10% of $B$) are interpretable in client terms. $\gamma_i$ is computed from SIM regression parameters and the regime-lens $\lambda$ (no news term). $\Sigma$ is the SIM decomposition $\sigma_m^2 \beta\beta^\top + \mathrm{diag}(\sigma_{\varepsilon,i}^2)$ — same parameters that drive $\gamma$.

> **MPC trigger conditions:**
>
> At decision time $t$, forward-project portfolio value $V_\tau$ for $\tau = t+1, \ldots, t+T$ by sampling $N = 1000$ paths from the SPY-JumpHMM marginal + SIM hybrid: $g_{i,\tau}^{(j)} = \alpha_i + \beta_i\, g_{m,\tau}^{(j)} + \varepsilon_{i,\tau}^{(j)}$. The realized path is **in-spec** at time $\tau$ iff
>
> $$\mu_\tau - z\,\sigma_\tau \;\le\; V_\tau \;\le\; \mu_\tau + z\,\sigma_\tau$$
>
> with defaults $z = 1.96$, $T = 21$ trading days. Re-allocation fires when **any** of: (1) $V_\tau$ exits the band on the realized path, (2) $T$ days have elapsed since the last allocation, (3) realized drawdown from peak exceeds $D_{\max} = 8\%$. Between triggers the engine submits no orders. This is the discipline that fixes the live failure mode.

### Algorithm: Constrained CD with MPC (Hold-Out Deployment Loop)

__Initialize__: Given the frozen 33-ticker basket $\mathcal{B}$, risk parameters $\sigma_{\max}$, $K_{\text{turnover}}$, $w_{\max}$, MPC parameters $z$, $T$, $D_{\max}$, and initial position $n_{\text{prev}} \gets 0$.

For each trading day $\tau$ in the hold-out window $[2025\text{-}01\text{-}02,\, 2026\text{-}04\text{-}22]$ __do__:

1. Compute $\gamma_\tau$ from SIM parameters, the EMA-based regime-lens $\lambda_\tau$, and the smoothed market growth $g_{m,\tau}$.
2. Forward-project $N$ paths over $[\tau, \tau + T]$ and form the in-spec band $[\mu_\tau - z\sigma_\tau,\; \mu_\tau + z\sigma_\tau]$.
3. Check the three trigger conditions on the current portfolio.
4. If any trigger fires, solve the constrained CD problem for $w_\tau^\star$; translate to integer shares $n_\tau$; route the order set through the cost + tax engine.
5. Update $n_{\text{prev}} \gets n_\tau$; log the trigger reason and the turnover consumed.
6. Otherwise hold; submit no orders.

__Output__: Wealth path $\{V_\tau\}$, trigger log, after-cost after-tax summary.

> **Baselines (five strategies compared against ConstrainedCDWithMPC):**
>
> All five use the same frozen 33-ticker basket and the same after-cost / after-tax engine; they differ only in the allocator and the rebalance cadence.
>
> **(1) EqualWeight (buy-and-hold):** $w_i = 1/K$ set once on day 1.
>
> **(2) MinVar (buy-and-hold):** solve
> $$\min_w \; w^\top \Sigma w \quad \text{s.t.} \quad \sum_i w_i = 1,\; w_i \ge 0$$
> on the training-window $\Sigma$, then hold. The S1 baseline.
>
> **(3) UnconstrainedCD (daily):** the closed-form Cobb-Douglas allocator. For preferred names ($\gamma_i > 0$):
> $$n_i = \frac{\gamma_i}{\sum_{j \in \text{pref}} \gamma_j} \cdot \frac{B_{\text{eff}}}{p_i}$$
> Non-preferred names ($\gamma_i \le 0$) pin at $n_i = \varepsilon = 10^{-3}$ shares. Rebalanced every trading day. This is the live engine's allocator at daily (not 30-min) cadence.
>
> **(4) CostAwareMV (daily):** the standard-finance alternative to constrained CD:
> $$\max_w \; \gamma^\top w \; - \; \tfrac{\kappa}{2}\, w^\top \Sigma w \; - \; c \,\| w - w_{\text{prev}} \|_1 \quad \text{s.t.} \quad \sum_i w_i = 1,\; w_i \ge 0$$
> Mean-variance with a turnover-cost penalty. Rebalanced daily.
>
> **(5) CDWithMPC:** the unconstrained CD formula from (3), but invoked **only when the MPC trigger fires** — same trigger logic as ConstrainedCDWithMPC. Isolates the cadence effect from the constraint effect.
>
> **(6) ConstrainedCDWithMPC (the design):** the full problem above, invoked on MPC trigger. Adds the covariance, turnover, and concentration constraints on top of (5).

Pairwise comparisons isolate effects: **(3) vs (5)** isolates the trigger-only fix (same allocator, different cadence); **(5) vs (6)** isolates the constraint-only fix (same trigger, more constraints); **(3) vs (6)** is the combined live-engine fix.

The implementation lives in the following scripts (the notebook only loads their saved results below):

- [`scripts/01_calibrate_sim.jl`](scripts/01_calibrate_sim.jl) → `sim_calibration.jld2`
- [`scripts/04_select_basket.jl`](scripts/04_select_basket.jl) → `frozen_basket.jld2` (with `03_train_bandit_mc.jl` upstream)
- [`scripts/06_backtest_mc.jl`](scripts/06_backtest_mc.jl) → `backtest_mc_results.jld2` (the notebook reads both `summary` and `per_seed_results`)

___
```

- [ ] **Step 3: Verify**

```bash
jq -r '.cells[1].source | join("") | .[0:200]' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```
Expected: starts with `___\n## Section 1: Theory Recap`.

---

### Task 8: Edit the include cell — no change (sanity check)

- [ ] **Step 1: Confirm cell 2 is `include("Include.jl")` and leave untouched**

```bash
jq -r '.cells[2] | {type: .cell_type, src: (.source | join(""))}' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: `type: "code"`, `src: "include(\"Include.jl\")"`. No edit; just verify.

---

### Task 9: Replace the artifact-load cell to drop `backtest_results.jld2` and compute canonical seed

**Files:**
- Modify: `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb` (cell 3, code)

- [ ] **Step 1: Locate cell 3's ID**

```bash
jq -r '.cells[3] | {id, type: .cell_type, head: (.source | join("") | .[0:60])}' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: code cell whose head begins with `# Load all artifacts`.

- [ ] **Step 2: Replace cell 3**

Use NotebookEdit `edit_mode = "replace"`, `cell_type = "code"`, `new_source`:

```julia
# Load all artifacts and compute the canonical reporting seed
_check_artifact(joinpath(_PATH_TO_ARTIFACTS, "sim_calibration.jld2"))
_check_artifact(joinpath(_PATH_TO_ARTIFACTS, "frozen_basket.jld2"))
_check_artifact(joinpath(_PATH_TO_ARTIFACTS, "backtest_mc_results.jld2"))

sim_calib = load_results(joinpath(_PATH_TO_ARTIFACTS, "sim_calibration.jld2"))
basket    = load_results(joinpath(_PATH_TO_ARTIFACTS, "frozen_basket.jld2"))
bt_mc     = load_results(joinpath(_PATH_TO_ARTIFACTS, "backtest_mc_results.jld2"))

n_seeds = bt_mc["config"]["n_seeds"]
sharpes = bt_mc["summary"]["ConstrainedCDWithMPCStrategy"]["sharpe_mc"]
seeds   = bt_mc["config"]["BACKTEST_MC_SEEDS"]
order   = sortperm(sharpes)
canonical_seed_idx = order[ceil(Int, length(order) / 2)]
canonical_seed     = seeds[canonical_seed_idx]
canonical          = bt_mc["per_seed_results"][canonical_seed_idx]

println("Hold-out: ", bt_mc["config"]["hold_out_start"],
        " to ", bt_mc["config"]["hold_out_end"],
        " (", bt_mc["config"]["n_days"], " days)")
println("Canonical reporting seed = ", canonical_seed,
        " (idx ", canonical_seed_idx, " of ", n_seeds,
        ", median Sharpe = ", round(median(sharpes); digits = 3), ")")
```

- [ ] **Step 3: Verify**

```bash
jq -r '.cells[3].source | join("") | .[0:120]' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```
Expected starts with `# Load all artifacts and compute the canonical reporting seed`.

---

### Task 10: Insert the Section 2 opener (Data Windows blockquote)

**Files:**
- Modify: `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb` (insert new markdown cell after cell 3)

- [ ] **Step 1: Note cell 3's ID (from Task 9 Step 1)**

The new cell will be inserted **after** cell 3 (the load cell).

- [ ] **Step 2: Insert the new Section 2 opener**

Use NotebookEdit `edit_mode = "insert"`, `cell_id = <cell 3's id>`, `cell_type = "markdown"`, `new_source`:

```markdown
___
## Section 2: Results

We compare the six strategies on the same 2025-2026 hold-out window using the same after-cost, after-tax engine. The constrained-CD design isolates against the live failure mode along two axes — cadence (clock vs. MPC trigger) and constraints (none vs. covariance+turnover+cap). Every metric below is computed on hold-out days; nothing is fit on this window.

> **Data windows:**
>
> - **Training:** 2014-01-03 to 2024-12-31, ~10 years of daily SPY-relative returns used to fit the per-ticker SIM parameters $(\alpha_i, \beta_i, \sigma_{\varepsilon,i})$.
> - **Hold-out:** 2025-01-02 to 2026-04-22, 326 trading days. Every strategy is forward-walked through this window with identical cost + tax rules.
> - **Universe:** 33-ticker basket frozen from the S4 per-sector bandit (median-Sharpe seed from the 30-seed run, uniform $q_s = 3$ per GICS sector). The universe does not change during the backtest.

The cell above loads the saved results and pins a **canonical reporting seed** — the seed whose `ConstrainedCDWithMPCStrategy` Sharpe equals the median of the 20-seed MC distribution. Single-seed displays below (wealth curves, trigger histogram) source from that seed's `per_seed_results` entry so they sit at a known location in the MC distribution rather than at an arbitrary index.
```

- [ ] **Step 3: Verify**

```bash
jq -r '.cells | length, (.cells[4].source | join("") | .[0:80])' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```
Expected: cell count increased to 16 (was 15), and `.cells[4].source` starts with `___\n## Section 2: Results`.

---

### Task 11: Insert the Frozen Basket subsection (markdown + code)

**Files:**
- Modify: `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb` (insert 2 new cells after cell 4)

- [ ] **Step 1: Find the ID of cell 4 (the Section 2 opener just inserted)**

```bash
jq -r '.cells[4] | {id, head: (.source | join("") | .[0:60])}' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

- [ ] **Step 2: Insert the Frozen Basket markdown cell**

NotebookEdit `edit_mode = "insert"`, `cell_id = <cell 4's id>`, `cell_type = "markdown"`, `new_source`:

```markdown
### Frozen Basket: Tickers and GICS Sectors

The S4 per-sector bandit was trained for 30 seeds; the median-Sharpe seed was frozen as the universe for every strategy below. The table groups the 33 tickers by GICS sector alongside the sector quota vector $(q_1, \ldots, q_{11})$ that the bandit was solved under (uniform $q_s = 3$). Sector tags come from `code/src/data/sp500-sectors.csv` via `load_sector_map`.
```

- [ ] **Step 3: Find the ID of the markdown cell just inserted**

```bash
jq -r '.cells[5] | {id, head: (.source | join("") | .[0:60])}' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: `head` starts with `### Frozen Basket`.

- [ ] **Step 4: Insert the Frozen Basket code cell**

NotebookEdit `edit_mode = "insert"`, `cell_id = <markdown cell ID from Step 3>`, `cell_type = "code"`, `new_source`:

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

- [ ] **Step 5: Verify**

```bash
jq -r '.cells | length, (.cells[5].cell_type), (.cells[6].cell_type), (.cells[6].source | join("") | .[0:60])' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```
Expected: cell count now 17, cell 5 type `markdown`, cell 6 type `code` starting with `sector_of, _ = load_sector_map`.

---

### Task 12: Edit the Bake-Off markdown + code cells (demote + new median-MC code)

**Files:**
- Modify: `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb` (cells 7 and 8 after the inserts above)

- [ ] **Step 1: Locate the bake-off cells**

```bash
jq -r '.cells[7], .cells[8] | {idx: input_filename, id, type: .cell_type, head: (.source | join("") | .[0:80])}' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: cell 7 markdown starting with `## Section 2: Headline Bake-Off`; cell 8 code starting with `rows = NamedTuple[]`.

- [ ] **Step 2: Replace cell 7 (markdown header demote + framing)**

NotebookEdit `edit_mode = "replace"`, `cell_id = <cell 7 id>`, `cell_type = "markdown"`, `new_source`:

```markdown
### Headline Bake-Off (after-cost, after-tax)

Single bake-off scorecard, sorted by hold-out Sharpe. Every figure is **median across the 20 MC seeds** — which collapses to the single value for the four deterministic strategies (EW, MinVar, UnconstrainedCD, CostAwareMV), and is the honest middle of the distribution for the two MPC strategies. To isolate effects, compare row-pairs from Theory Blockquote 3: **(3) vs (5)** for trigger-only, **(5) vs (6)** for constraint-only, **(3) vs (6)** for the combined live-engine fix.
```

- [ ] **Step 3: Replace cell 8 (bake-off code with median-MC sourcing)**

NotebookEdit `edit_mode = "replace"`, `cell_id = <cell 8 id>`, `cell_type = "code"`, `new_source`:

```julia
strat_names = sort(collect(keys(bt_mc["per_seed_results"][1])))
rows = NamedTuple[]
for name in strat_names
    sharpes_n = [bt_mc["per_seed_results"][i][name].summary.ann_sharpe     for i in 1:n_seeds]
    rets_n    = [bt_mc["per_seed_results"][i][name].summary.ann_return     for i in 1:n_seeds]
    dds_n     = [bt_mc["per_seed_results"][i][name].summary.max_drawdown   for i in 1:n_seeds]
    turns_n   = [bt_mc["per_seed_results"][i][name].summary.ann_turnover   for i in 1:n_seeds]
    ntrigs_n  = [bt_mc["per_seed_results"][i][name].summary.n_mpc_triggers for i in 1:n_seeds]
    push!(rows, (Strategy = name,
        Sharpe_med     = round(median(sharpes_n); digits = 3),
        AnnRet_med_pct = round(median(rets_n) * 100; digits = 2),
        MaxDD_med_pct  = round(median(dds_n)  * 100; digits = 1),
        Turn_med       = round(median(turns_n); digits = 3),
        N_trig_med     = round(Int, median(ntrigs_n))))
end
sort!(rows; by = r -> -r.Sharpe_med)
pretty_table(DataFrame(rows); backend = :text)
```

- [ ] **Step 4: Verify**

```bash
jq -r '.cells[7].source | join("") | .[0:60]' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
jq -r '.cells[8].source | join("") | .[0:60]' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```
Expected: cell 7 starts with `### Headline Bake-Off`; cell 8 starts with `strat_names = sort(collect(keys(bt_mc`.

---

### Task 13: Edit the Wealth Curves cells (demote + canonical-seed source)

**Files:**
- Modify: `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb` (cells 9 and 10)

- [ ] **Step 1: Locate the cells**

```bash
jq -r '.cells[9], .cells[10] | {id, type: .cell_type, head: (.source | join("") | .[0:60])}' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: cell 9 markdown `## Section 3: Wealth Curves`; cell 10 code starting with `p = plot(...)`.

- [ ] **Step 2: Replace cell 9 markdown**

NotebookEdit `replace`, `cell_id = <cell 9 id>`, `cell_type = "markdown"`, `new_source`:

```markdown
### Wealth Curves

After-cost, after-tax wealth paths for the canonical seed (idx pinned in the load cell). The four deterministic strategies give the same path on every seed; the two MPC strategies show their median-Sharpe seed's path so the wealth curve sits at a known location in the MC distribution.
```

- [ ] **Step 3: Replace cell 10 code (source from canonical)**

NotebookEdit `replace`, `cell_id = <cell 10 id>`, `cell_type = "code"`, `new_source`:

```julia
p = plot(legend = :outerright, size = (1080, 540),
         xlabel = "Trading day", ylabel = "Wealth (after-cost, after-tax)")
for (name, r) in canonical
    plot!(p, r.wealth_after_cost_aftertax; label = name, lw = 1.4)
end
p
```

- [ ] **Step 4: Verify** with the same jq commands as Step 1. Cell 9 head: `### Wealth Curves`; cell 10 head: `p = plot(legend`.

---

### Task 14: Edit the MPC Trigger Reasons cells (demote + canonical-seed source)

**Files:**
- Modify: `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb` (cells 11 and 12)

- [ ] **Step 1: Locate cells**

```bash
jq -r '.cells[11], .cells[12] | {id, type: .cell_type, head: (.source | join("") | .[0:60])}' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: cell 11 markdown `## Section 4: MPC Trigger Reasons`; cell 12 code iterating `bt["results"]`.

- [ ] **Step 2: Replace cell 11 markdown**

NotebookEdit `replace`, `cell_id = <cell 11 id>`, `cell_type = "markdown"`, `new_source`:

```markdown
### MPC Trigger Reasons

Only the two MPC strategies (CDWithMPC, ConstrainedCDWithMPC) maintain a trigger log. Each fire is tagged with the condition that tripped it — out-of-band, $T$-day refresh, or the drawdown circuit-breaker — so we can see whether re-allocations are driven by realized drift or by the calendar. Counts shown are for the canonical reporting seed.
```

- [ ] **Step 3: Replace cell 12 code**

NotebookEdit `replace`, `cell_id = <cell 12 id>`, `cell_type = "code"`, `new_source`:

```julia
for (name, r) in canonical
    if !isempty(r.trigger_log)
        reasons = [t.reason for t in r.trigger_log if t.fired]
        if !isempty(reasons)
            counts = Dict(rs => count(==(rs), reasons) for rs in unique(reasons))
            println(rpad(name, 35), "  ", counts)
        end
    end
end
```

- [ ] **Step 4: Verify** as in prior tasks.

---

### Task 15: Edit the Multi-Seed Backtest markdown (demote + keep prose)

**Files:**
- Modify: `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb` (cell 13)

- [ ] **Step 1: Locate**

```bash
jq -r '.cells[13] | {id, type: .cell_type, head: (.source | join("") | .[0:80])}' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: markdown `## Section 5: Multi-Seed Backtest Distribution`.

- [ ] **Step 2: Replace cell 13 markdown (demote, keep most existing prose, retarget to MC table)**

NotebookEdit `replace`, `cell_id = <cell 13 id>`, `cell_type = "markdown"`, `new_source`:

```markdown
### Multi-Seed Backtest Distribution

The two MPC strategies depend on `BACKTEST_RNG_SEED` through their `forward_project` Monte Carlo paths. To honestly report performance we run each strategy across 20 seeds (`2001:2020`) and report the distribution of outcomes. The four non-MPC strategies (EW, MinVar, UnconstrainedCD, CostAwareMV) are deterministic given prices and collapse to a single value — they are included in the table to keep the row set uniform.
```

- [ ] **Step 3: Verify cell 14 (the MC summary code) is unchanged**

```bash
jq -r '.cells[14] | {type: .cell_type, head: (.source | join("") | .[0:80])}' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: code cell starting with `_check_artifact(joinpath(_PATH_TO_ARTIFACTS, "backtest_mc_results.jld2"))` OR (if the cell already had its load lines removed) starting with `rows = NamedTuple[]`. **If the cell still calls `_check_artifact(... backtest_mc_results.jld2)` and re-assigns `bt_mc` and `n_seeds`, replace it with the simpler form** to avoid double-loading (since Task 9's consolidated load cell already does this):

NotebookEdit `replace`, `cell_id = <cell 14 id>`, `cell_type = "code"`, `new_source`:

```julia
rows = NamedTuple[]
for (name, agg) in bt_mc["summary"]
    sh = agg["sharpe_mc"]
    dd = agg["max_dd_mc"]
    wt = agg["W_T_over_W0_mc"]
    push!(rows, (Strategy = name,
        Sharpe_min    = round(minimum(sh); digits = 3),
        Sharpe_med    = round(median(sh);  digits = 3),
        Sharpe_max    = round(maximum(sh); digits = 3),
        MaxDD_med_pct = round(median(dd) * 100; digits = 1),
        WT_W0_med     = round(median(wt); digits = 3),
        nTrig_med     = round(Int, median(agg["n_mpc_triggers_mc"]))))
end
sort!(rows; by = r -> -r.Sharpe_med)
pretty_table(DataFrame(rows); backend = :text)
```

---

### Task 16: Demote Sharpe histogram header `### → ####`

**Files:**
- Modify: `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb` (cell 15)

- [ ] **Step 1: Locate**

```bash
jq -r '.cells[15] | {id, type: .cell_type, head: (.source | join("") | .[0:80])}' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: markdown starting with `### Sharpe distribution histogram`.

- [ ] **Step 2: Replace with demoted heading + prose**

NotebookEdit `replace`, `cell_id = <cell 15 id>`, `cell_type = "markdown"`, `new_source`:

```markdown
#### Sharpe distribution histogram — ConstrainedCDWithMPCStrategy

Hold-out Sharpe distribution for the constrained-CD + MPC design across the 20 backtest seeds. The four deterministic strategies (EW, MinVar, UnconstrainedCD, CostAwareMV) collapse to a single point — they do not vary with the RNG.
```

- [ ] **Step 3: Verify cell 16 (histogram code) starts with `sh = bt_mc["summary"]["ConstrainedCDWithMPCStrategy"]`**

```bash
jq -r '.cells[16].source | join("") | .[0:80]' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```
No edit if it matches; otherwise replace with the existing histogram body.

---

### Task 17: Replace top-level Disclaimer with the Summary cell (placeholders intact)

**Files:**
- Modify: `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb` (cell 17 — the final cell)

- [ ] **Step 1: Locate the last cell**

```bash
jq -r '.cells | length, (.cells[-1] | {id, type: .cell_type, head: (.source | join("") | .[0:80])})' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: cell count 18, last cell markdown starting with `## Disclaimer`.

- [ ] **Step 2: Replace the final cell with Summary + Disclaimer**

NotebookEdit `replace`, `cell_id = <last cell id>`, `cell_type = "markdown"`, `new_source`:

```markdown
___
## Summary

The ConstrainedCD+MPC design replaces the live engine's failure mode — unconstrained Cobb-Douglas fired on a 30-min clock with the turnover gate disabled — with two disciplined components: a constrained allocator that respects a covariance budget, a turnover budget, and a concentration cap, and an MPC trigger that fires re-allocation only when the realized portfolio path leaves a forward-projection band. On the 2025-2026 hold-out, after costs and lot-by-lot FIFO taxes, ConstrainedCDWithMPC delivers a hold-out Sharpe of {S_6} on the 33-ticker frozen basket, against {S_3} for UnconstrainedCD-daily and {S_1} for equal-weight, while running at roughly {T_ratio}× lower annualized turnover than the daily-rebalance strategies.

> **Key Takeaways:**
>
> - **Constraints and cadence are both load-bearing:** The pairwise comparison isolates each effect. Going UnconstrainedCD-daily (3) → CDWithMPC (5) is the trigger-only fix; CDWithMPC (5) → ConstrainedCDWithMPC (6) is the constraint-only fix. Max-drawdown drops by {ΔDD_53} pp and {ΔDD_65} pp respectively; together they account for the full move from the live engine's flat-with-bleed P&L to the design's hold-out performance. Neither half alone closes the gap.
>
> - **MPC trigger reasons concentrate on band exits:** The trigger log shows that re-allocation is dominated by the realized path leaving the forward-projection band, not by the $T = 21$ calendar refresh or the drawdown circuit-breaker. The engine fires {N_trig} times across 326 hold-out days at the canonical seed — roughly an order of magnitude fewer order events than a daily-rebalance baseline, and zero events on a 30-min clock for free.
>
> - **Read distributions, not single trials:** The two MPC strategies are stochastic in `BACKTEST_RNG_SEED` through their forward-projection MC paths. The 20-seed Sharpe distribution shows a standard deviation of {σ_Sharpe} for ConstrainedCDWithMPC; the median is the honest summary statistic, and reporting only one seed misrepresents the strategy. The four deterministic baselines (EW, MinVar, UnconstrainedCD, CostAwareMV) collapse to a single point per metric — directly comparable to the MC median, not to any individual seed.

This closes the loop opened by the live-engine post-mortem in `constrained_cobb_douglas.md`. The original engine had no notion of cost, no notion of risk, no notion of holding, and fired on a clock with no information content. The constrained CD allocator gives it the first three; the MPC discipline gives it the fourth.

### Disclaimer

This content is for educational purposes only and does not constitute investment advice. The examples use real historical data, a frozen SIM calibration on 2014-2024, and a single 2025-2026 forward window; conclusions about cost-aware constrained allocation and MPC trigger discipline do not generalize to other markets, time periods, or client risk profiles without re-calibration.

___
```

The `{S_6}`, `{S_3}`, `{S_1}`, `{T_ratio}`, `{ΔDD_53}`, `{ΔDD_65}`, `{N_trig}`, `{σ_Sharpe}` placeholders are filled in Task 19.

- [ ] **Step 3: Verify**

```bash
jq -r '.cells | length, (.cells[-1].source | join("") | .[0:120])' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```
Expected: cell count still 18, last cell starts with `___\n## Summary\n\nThe ConstrainedCD+MPC design`.

---

### Task 18: Execute the notebook end-to-end

**Files:**
- Read/modify: `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb` (writes back with cell outputs)

- [ ] **Step 1: Run `jupyter nbconvert --execute --inplace`**

```bash
cd /Users/jdv27/Desktop/julia_work/modified_cobb_douglas_example && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: exit code 0; no exception in any cell. If any cell errors, STOP and debug — likely a typo in the code cell replacements or a stale variable from a removed cell.

- [ ] **Step 2: Spot-check the executed outputs**

```bash
jq -r '.cells | map(select(.cell_type == "code")) | .[] | {head: (.source | join("") | .[0:50]), n_outputs: (.outputs | length)}' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: every code cell has at least one output entry (except possibly the `include("Include.jl")` cell, which may have only setup output). The roster cell should have a stream output with 33 rows.

---

### Task 19: Compute and substitute Summary placeholders

**Files:**
- Read: `scripts/data/backtest_mc_results.jld2`
- Modify: `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb` (Summary cell)

- [ ] **Step 1: Compute the placeholder values**

```bash
julia --project=code -e '
using JLD2, Statistics, Printf
bt = JLD2.load("scripts/data/backtest_mc_results.jld2")
sm = bt["summary"]

S_6 = median(sm["ConstrainedCDWithMPCStrategy"]["sharpe_mc"])
S_3 = median(sm["UnconstrainedCDStrategy"]["sharpe_mc"])
S_1 = median(sm["EqualWeightStrategy"]["sharpe_mc"])

T_ratio = median(sm["UnconstrainedCDStrategy"]["ann_turnover_mc"]) /
          median(sm["ConstrainedCDWithMPCStrategy"]["ann_turnover_mc"])

DD_3 = median(sm["UnconstrainedCDStrategy"]["max_dd_mc"]) * 100
DD_5 = median(sm["CDWithMPCStrategy"]["max_dd_mc"]) * 100
DD_6 = median(sm["ConstrainedCDWithMPCStrategy"]["max_dd_mc"]) * 100
ΔDD_53 = DD_3 - DD_5
ΔDD_65 = DD_5 - DD_6

N_trig = round(Int, median(sm["ConstrainedCDWithMPCStrategy"]["n_mpc_triggers_mc"]))
σ_Sharpe = std(sm["ConstrainedCDWithMPCStrategy"]["sharpe_mc"])

@printf("S_6=%.3f S_3=%.3f S_1=%.3f T_ratio=%.1f\n", S_6, S_3, S_1, T_ratio)
@printf("DDelta_53=%.1f DDelta_65=%.1f N_trig=%d sigma_Sharpe=%.3f\n", ΔDD_53, ΔDD_65, N_trig, σ_Sharpe)
'
```

Capture the eight printed numbers.

- [ ] **Step 2: Substitute placeholders in the Summary cell**

For each placeholder in the Summary cell, use the Edit tool with `replace_all=false` against the notebook .ipynb file (Edit is fine on JSON — the placeholder strings are unique). Substitute:

- `{S_6}` → the rounded Sharpe (3 digits, e.g. `1.234`)
- `{S_3}` → ditto
- `{S_1}` → ditto
- `{T_ratio}` → 1 digit (e.g. `5.7`)
- `{ΔDD_53}` → 1 digit (e.g. `2.1`)
- `{ΔDD_65}` → ditto
- `{N_trig}` → integer (no decimals)
- `{σ_Sharpe}` → 3 digits (e.g. `0.142`)

- [ ] **Step 3: Verify no placeholders remain**

```bash
grep -E '\{S_[613]\}|\{T_ratio\}|\{ΔDD_5?3?5?\}|\{N_trig\}|\{σ_Sharpe\}' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: no matches (empty output, exit 1 from grep is fine).

- [ ] **Step 4: Re-execute the notebook to refresh the cached cell with the new text**

Same command as Task 18 Step 1. Confirm exit 0.

---

### Task 20: Final acceptance check + commit Phase B

**Files:**
- Read: `eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb`

- [ ] **Step 1: Verify the four-section structure (spec §5 acceptance)**

```bash
jq -r '.cells | map(select(.cell_type == "markdown")) | map(.source | join("") | split("\n")[0]) | map(select(startswith("#"))) | .[]' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected output (in order):
```
# Constrained Cobb-Douglas with MPC — Theory and Hold-Out Results
___
## Section 1: Theory Recap
### Algorithm: Constrained CD with MPC (Hold-Out Deployment Loop)
___
## Section 2: Results
### Frozen Basket: Tickers and GICS Sectors
### Headline Bake-Off (after-cost, after-tax)
### Wealth Curves
### MPC Trigger Reasons
### Multi-Seed Backtest Distribution
#### Sharpe distribution histogram — ConstrainedCDWithMPCStrategy
___
## Summary
### Disclaimer
```

(The `___` lines are markdown horizontal rules, separators in our cells; they may appear depending on how the split sees them. The key checks are: exactly three `##` lines — Section 1, Section 2, Summary — and `### Disclaimer` is `###` not `##`.)

- [ ] **Step 2: Confirm no `## Disclaimer` exists at top level**

```bash
jq -r '.cells | map(.source | join("")) | .[] | select(startswith("## Disclaimer"))' eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Expected: empty output.

- [ ] **Step 3: Commit the notebook reformat**

```bash
git add eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
git commit -m "$(cat <<'EOF'
S5 notebook: reformat to four-section layout (Intro/Theory/Results/Summary)

Reformats the constrained Cobb-Douglas notebook to match the S4 bandit
example's section structure. Replaces the stubbed Theory Recap with an
inline recap of the constrained CD optimization, MPC trigger conditions,
the hold-out deployment algorithm, and the five baseline allocator
equations. Adds a Frozen Basket roster showing the 33 tickers grouped
by GICS sector. Drops the dependency on the redundant backtest_results
artifact; bake-off, wealth curves, and trigger reasons now source from
backtest_mc_results at the median-Sharpe canonical seed. Adds a Summary
section with headline paragraph, Key Takeaways blockquote, and demoted
Disclaimer.

Spec: docs/superpowers/specs/2026-05-18-s5-notebook-reformat-design.md
Plan: docs/superpowers/plans/2026-05-18-s5-notebook-reformat.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist (to run after plan execution)

After all tasks complete, confirm spec §5 acceptance criteria:

1. Three top-level `##` headers — Section 1, Section 2, Summary. ✓ (Task 20 Step 1)
2. `###` subsections under Results in order: Frozen Basket, Headline Bake-Off, Wealth Curves, MPC Trigger Reasons, Multi-Seed Backtest Distribution. Sharpe histogram is `####`. ✓ (Task 20 Step 1)
3. No top-level Disclaimer. ✓ (Task 20 Step 2)
4. Notebook executes end-to-end. ✓ (Task 18, re-confirmed in Task 19 Step 4)
5. All `{...}` placeholders resolved. ✓ (Task 19 Step 3)
6. Theory Recap is self-contained. ✓ (Task 7 — no "see spec doc X" stubs)
