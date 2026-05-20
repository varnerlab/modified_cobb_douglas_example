# Modified Cobb-Douglas Example

[![Documentation](https://github.com/varnerlab/modified_cobb_douglas_example/actions/workflows/docs.yml/badge.svg)](https://varnerlab.org/modified_cobb_douglas_example/dev/)

A teaching repository for the eCornell *AI in Finance* Session 5 example: a **constrained Cobb-Douglas portfolio allocator** wrapped in **model-predictive-control (MPC)** discipline, backtested against five baseline strategies on a 2025-2026 hold-out window of the S&P 500.

The economic objective is the familiar log-utility Cobb-Douglas form — the investor maximizes Σ γᵢ log(nᵢ) over a curated basket — but the allocator now respects a covariance budget, a turnover budget, and a per-name concentration cap, and the engine re-decides only when the realized portfolio path leaves a forward-projected confidence band (the MPC trigger) rather than on a fixed clock. The universe is selected by a per-sector ε-greedy bandit trained on 2014-2024 returns, and the backtest harness compares six strategies head-to-head with shared cost, tax, and execution accounting.

## What's in here

- [`code/`](code) — the `ConstrainedCobbDouglas.jl` package: SIM calibration, the constrained CD solver and four baselines, MPC trigger, cost model, tax ledger, per-sector bandit, and the backtest harness.
- [`scripts/`](scripts) — the end-to-end pipeline (`01_calibrate_sim.jl` through `06_backtest_mc.jl`) plus the four parameter sweeps (`07_sigma_max_sweep.jl`, `08_w_max_sweep.jl`, `09_k_turnover_sweep.jl`, `10_cash_revisit_sweep.jl`).
- [`eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb`](eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb) — the Session 5 notebook. It does no compute; it loads the JLD2 artifacts written by `scripts/01-05` and renders the headline table and wealth curves.
- [`docs/`](docs) — Documenter.jl source for the API and theory documentation deployed via GitHub Actions.
- [`constrained_cobb_douglas.md`](constrained_cobb_douglas.md) — the strategy design document with the full derivation, constraints, and motivation.

## Documentation

Full API reference, theory walk-through, and usage guides:

**https://varnerlab.org/modified_cobb_douglas_example/dev/**

## Installation

1. Install Julia 1.10 or later from [julialang.org/downloads](https://julialang.org/downloads/).
2. Clone this repository:
   ```bash
   git clone https://github.com/varnerlab/modified_cobb_douglas_example.git
   cd modified_cobb_douglas_example
   ```
3. Instantiate the package environment. From the repo root:
   ```bash
   julia --project=code -e 'using Pkg; Pkg.instantiate()'
   ```
   This resolves all dependencies listed in [`code/Project.toml`](code/Project.toml), including the unregistered `JumpHMM.jl` pulled from the URL declared in the `[sources]` block.
4. (Optional, for the notebook) install Jupyter and the `IJulia` kernel:
   ```bash
   julia --project=code -e 'using Pkg; Pkg.add("IJulia")'
   ```

## Running the example

Run the pipeline from the repo root so the script-relative artifact paths resolve:

```bash
julia --project=code scripts/01_calibrate_sim.jl
julia --project=code scripts/02_train_bandit.jl
julia --project=code scripts/03_train_bandit_mc.jl
julia --project=code scripts/04_select_basket.jl
julia --project=code scripts/05_backtest_strategies.jl
```

Each script writes a single JLD2 file to `scripts/data/`. Then open the notebook to render the results:

```bash
jupyter notebook eCornell-AI-Finance-S5-Example-ConstrainedCobbDouglas-May-2026.ipynb
```

Scripts `06` through `10` produce the Monte-Carlo replicate and the four robustness sweeps; they consume the artifacts from `01-05` and are optional for the headline result.

## Disclaimer

This repository is an **educational example** that accompanies the eCornell *AI in Finance* Session 5 lecture. It is provided **for instructional and research purposes only** and is **not investment advice, a solicitation, or a recommendation** to buy, sell, or hold any security. The constrained Cobb-Douglas + MPC framework, the bandit-selected basket, and every backtest result shipped in the JLD2 artifacts are illustrative; they are not a guaranteed-return product and they have not been audited, certified, or vetted for production use. Past backtested performance does not guarantee future results, and the cost, tax, slippage, and liquidity assumptions baked into the harness are deliberately simplified for teaching.

The authors are not licensed investment professionals. Anyone considering deploying any portion of this code with real capital should perform their own independent due diligence, consult a licensed financial advisor, and verify the modeling assumptions, parameters, and risk controls against their own risk tolerance and regulatory context. The code is distributed **as-is**, with no warranty of any kind, under the terms of the [MIT License](LICENSE); the authors and the Varner Lab accept no liability for any loss, direct or indirect, arising from its use.

## License

[MIT](LICENSE).
