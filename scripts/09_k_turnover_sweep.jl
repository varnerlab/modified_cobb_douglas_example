# scripts/09_k_turnover_sweep.jl
# Single-variable sensitivity sweep: ConstrainedCDWithMPCStrategy at each
# K_turnover in a grid, 20-seed MC each. σ_max is pinned at 10.0 (cov inactive)
# and w_max at 0.15 (the new best from the 08 sweep). Everything else mirrors
# 06/07/08. Writes scripts/data/k_turnover_sweep.jld2.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "code"))

using ConstrainedCobbDouglas
using JLD2
using JumpHMM
using Statistics
using Dates
using Printf

const PATH_INPUTS = joinpath(@__DIR__, "..", "code", "src", "data")
const PATH_OUT    = joinpath(@__DIR__, "data")
const BACKTEST_MC_SEEDS = 2001:2020
const B_0 = 100_000.0
const SIGMA_MAX_FIXED = 10.0
const W_MAX_FIXED     = 0.15
const K_TURNOVER_MULTIPLIERS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 10.0]
const K_TURNOVER_GRID = K_TURNOVER_MULTIPLIERS .* B_0

println("=" ^ 78)
println("09_k_turnover_sweep.jl — K_turnover sensitivity sweep, ConstrainedCDWithMPC")
println("=" ^ 78)

# --- Mirror 06 environment setup ---------------------------------------------
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
dates_hold = vcat([Date(d) for d in ohlc_2025.dates],
                  [Date(d) for d in ohlc_2026.dates])
n_days = size(prices_hold, 1)
println("Hold-out: $n_days days, $(length(basket_tickers)) tickers, ",
        length(K_TURNOVER_GRID), " K_turnover values, ",
        length(BACKTEST_MC_SEEDS), " seeds")
println("Fixed: σ_max = $SIGMA_MAX_FIXED, w_max = $W_MAX_FIXED")

spy_2025_idx = findfirst(==("SPY"), all_tickers_2025)
spy_2026_idx = findfirst(==("SPY"), all_tickers_2026)
if spy_2025_idx !== nothing && spy_2026_idx !== nothing
    market_prices = vcat(Vector{Float64}(ohlc_2025.prices[:, spy_2025_idx]),
                         Vector{Float64}(ohlc_2026.prices[:, spy_2026_idx]))
else
    market_prices = vec(mean(prices_hold; dims = 2))
end

sim_calib_tickers = sim_calib["tickers"]
sim_col = Dict(t => i for (i, t) in enumerate(sim_calib_tickers))
αs   = Float64[sim_calib["alpha"][sim_col[t]]     for t in basket_tickers]
βs   = Float64[sim_calib["beta"][sim_col[t]]      for t in basket_tickers]
σ_εs = Float64[sim_calib["sigma_eps"][sim_col[t]] for t in basket_tickers]
σ_m  = Float64(sim_calib["sigma_market"])

sim_init = Dict(basket_tickers[i] => ewls_init(αs[i], βs[i], σ_εs[i];
    half_life = 252.0, prior_weight = 252.0) for i in eachindex(basket_tickers))

market_model = load(joinpath(PATH_INPUTS, "pretrained-jumphmm-market-surrogate.jld2"),
    "model")

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

spec = MyMPCSpec(z = 1.96, T = 21, N = 1000, D_max = 0.08)

# --- Sweep -------------------------------------------------------------------
n_seeds = length(BACKTEST_MC_SEEDS)
n_kt    = length(K_TURNOVER_GRID)
println("\nRunning ConstrainedCDWithMPC at $n_kt K_turnover × $n_seeds seeds = ",
        n_kt * n_seeds, " backtests\n")

per_kt_results = Dict{Float64,Vector{MyBacktestResult}}()
summary = Dict{Float64,Dict{String,Any}}()

t_start = time()
for (j, kt) in enumerate(K_TURNOVER_GRID)
    strat = ConstrainedCDWithMPCStrategy(spec = spec,
        σ_max = SIGMA_MAX_FIXED, K_turnover = kt, w_max = W_MAX_FIXED)

    results = Vector{MyBacktestResult}(undef, n_seeds)
    for (i, s) in enumerate(BACKTEST_MC_SEEDS)
        results[i] = run_backtest(strat, env, cost_model, tax_rates;
                                  B₀ = B_0, rng_seed = s)
    end
    per_kt_results[kt] = results

    sharpe_mc       = Float64[r.summary.ann_sharpe       for r in results]
    max_dd_mc       = Float64[r.summary.max_drawdown     for r in results]
    ann_turnover_mc = Float64[r.summary.ann_turnover     for r in results]
    ann_return_mc   = Float64[r.summary.ann_return       for r in results]
    n_trig_mc       = Int[r.summary.n_mpc_triggers       for r in results]
    W_T_over_W0_mc  = Float64[r.wealth_after_cost_pretax[end] /
                              r.wealth_after_cost_pretax[1] for r in results]

    reason_counts = Dict(:band_exit => 0, :horizon_elapsed => 0, :drawdown => 0)
    for r in results, t in r.trigger_log
        t.fired || continue
        reason_counts[t.reason] = get(reason_counts, t.reason, 0) + 1
    end
    summary[kt] = Dict{String,Any}(
        "sharpe_mc"               => sharpe_mc,
        "max_dd_mc"               => max_dd_mc,
        "ann_turnover_mc"         => ann_turnover_mc,
        "ann_return_mc"           => ann_return_mc,
        "n_mpc_triggers_mc"       => n_trig_mc,
        "W_T_over_W0_mc"          => W_T_over_W0_mc,
        "trigger_reason_total"    => Dict(String(k) => v for (k, v) in reason_counts),
        "trigger_reason_per_seed" => Dict(String(k) => v / n_seeds
                                          for (k, v) in reason_counts))

    @printf("K_turn=\$%9.0f (%4.2f·B0, %d/%d):  Sharpe med=%.3f  IQR=[%.3f,%.3f]  MaxDD%% med=%.1f  trigs/seed med=%d\n",
            kt, kt / B_0, j, n_kt,
            median(sharpe_mc), quantile(sharpe_mc, 0.25), quantile(sharpe_mc, 0.75),
            median(max_dd_mc) * 100, Int(round(median(n_trig_mc))))
end
t_elapsed = time() - t_start
@printf("\nWall-clock: %.1f s (%.2f s per K_turnover × %d seeds)\n",
        t_elapsed, t_elapsed / n_kt, n_seeds)

# --- Summary table -----------------------------------------------------------
println("\nK_turnover sweep summary (median across $n_seeds seeds):")
println("-" ^ 110)
@printf("%-12s  %-7s  %8s %8s %8s   %8s %8s %8s   %8s  %5s %5s %5s\n",
        "K_turn (\$)", "x B0",
        "Shp_Q25", "Shp_med", "Shp_Q75",
        "DD_min%", "DD_med%", "DD_max%",
        "Turn_med", "band", "dd", "horz")
println("-" ^ 110)
for kt in K_TURNOVER_GRID
    s = summary[kt]
    sh, dd, tn = s["sharpe_mc"], s["max_dd_mc"], s["ann_turnover_mc"]
    rps = s["trigger_reason_per_seed"]
    @printf("%12.0f  %7.2f  %8.3f %8.3f %8.3f   %8.1f %8.1f %8.1f   %8.3f  %5.1f %5.1f %5.1f\n",
            kt, kt / B_0,
            quantile(sh, 0.25), median(sh), quantile(sh, 0.75),
            minimum(dd) * 100, median(dd) * 100, maximum(dd) * 100,
            median(tn),
            get(rps, "band_exit", 0.0),
            get(rps, "drawdown", 0.0),
            get(rps, "horizon_elapsed", 0.0))
end
println("-" ^ 110)

# --- Persist -----------------------------------------------------------------
save_results(joinpath(PATH_OUT, "k_turnover_sweep.jld2"), Dict(
    "config" => Dict(
        "BACKTEST_MC_SEEDS"     => collect(BACKTEST_MC_SEEDS),
        "n_seeds"               => n_seeds,
        "k_turnover_grid"       => K_TURNOVER_GRID,
        "k_turnover_multipliers"=> K_TURNOVER_MULTIPLIERS,
        "sigma_max_fixed"       => SIGMA_MAX_FIXED,
        "w_max_fixed"           => W_MAX_FIXED,
        "hold_out_start"        => string(dates_hold[1]),
        "hold_out_end"          => string(dates_hold[end]),
        "n_days"                => n_days,
        "K"                     => length(basket_tickers),
        "tickers"               => basket_tickers,
        "B_0"                   => B_0,
        "tax_rates"             => Dict("st" => tax_rates.st, "lt" => tax_rates.lt),
        "ewls_half_life_days"   => 252,
        "MPC_spec"              => Dict("z" => spec.z, "T" => spec.T,
                                        "N" => spec.N, "D_max" => spec.D_max)),
    "k_turnover_grid"   => K_TURNOVER_GRID,
    "summary"           => summary,
    "per_kt_results"    => per_kt_results))
println("\nSaved scripts/data/k_turnover_sweep.jld2")
