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
