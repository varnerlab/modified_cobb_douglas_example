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
