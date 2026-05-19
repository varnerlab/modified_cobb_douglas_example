# scripts/10_cash_revisit_sweep.jl
# Single-variable sensitivity sweep: ConstrainedCDWithMPCStrategy at each
# cash_revisit_interval in a grid, 20-seed MC each. σ_max is pinned at 10.0
# (cov inactive), w_max at 0.20 (bake-off default), and K_turnover at the
# 0.10·B_0 = $10,000 baseline. Everything else mirrors 06/07/08/09.
#
# The cash_revisit_interval gates the new fourth trigger condition: while the
# allocator is sitting in the ε-pin defensive regime (γ_i ≤ 0 across the
# basket), the strategy re-evaluates after this many trading days instead of
# waiting the full T-day horizon. Default = T = 21 preserves the original
# behavior; smaller values let the strategy re-enter the market sooner after
# a defensive fire. Canonical-seed smoke testing suggested interval = 5 is a
# Sharpe sweet spot, but the response is highly non-monotonic, so a proper
# 20-seed sweep is required before drawing conclusions.
#
# Writes scripts/data/cash_revisit_sweep.jld2.

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
const SIGMA_MAX_FIXED  = 10.0
const W_MAX_FIXED      = 0.20
const K_TURNOVER_FIXED = 0.10 * B_0   # $10,000 per decision (bake-off default)
const CASH_REVISIT_GRID = [1, 3, 5, 7, 10, 14, 21]

println("=" ^ 78)
println("10_cash_revisit_sweep.jl — cash_revisit_interval sweep, ConstrainedCDWithMPC")
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
        length(CASH_REVISIT_GRID), " interval values, ",
        length(BACKTEST_MC_SEEDS), " seeds")
println("Fixed: σ_max = $SIGMA_MAX_FIXED, w_max = $W_MAX_FIXED, ",
        "K_turnover = \$$(Int(K_TURNOVER_FIXED))")

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

# --- Sweep -------------------------------------------------------------------
n_seeds    = length(BACKTEST_MC_SEEDS)
n_interval = length(CASH_REVISIT_GRID)
println("\nRunning ConstrainedCDWithMPC at $n_interval intervals × $n_seeds seeds = ",
        n_interval * n_seeds, " backtests\n")

per_interval_results = Dict{Int,Vector{MyBacktestResult}}()
summary              = Dict{Int,Dict{String,Any}}()

t_start = time()
for (j, interval) in enumerate(CASH_REVISIT_GRID)
    # Build a fresh spec for each grid point — only cash_revisit_interval varies.
    spec = MyMPCSpec(z = 1.96, T = 21, N = 1000, D_max = 0.08,
                     cash_revisit_interval = interval)
    strat = ConstrainedCDWithMPCStrategy(spec = spec,
        σ_max = SIGMA_MAX_FIXED, K_turnover = K_TURNOVER_FIXED, w_max = W_MAX_FIXED)

    results = Vector{MyBacktestResult}(undef, n_seeds)
    for (i, s) in enumerate(BACKTEST_MC_SEEDS)
        results[i] = run_backtest(strat, env, cost_model, tax_rates;
                                  B₀ = B_0, rng_seed = s)
    end
    per_interval_results[interval] = results

    sharpe_mc       = Float64[r.summary.ann_sharpe       for r in results]
    max_dd_mc       = Float64[r.summary.max_drawdown     for r in results]
    ann_turnover_mc = Float64[r.summary.ann_turnover     for r in results]
    ann_return_mc   = Float64[r.summary.ann_return       for r in results]
    n_trig_mc       = Int[r.summary.n_mpc_triggers       for r in results]
    W_T_over_W0_mc  = Float64[r.wealth_after_cost_pretax[end] /
                              r.wealth_after_cost_pretax[1] for r in results]

    # Initialize all four trigger reasons so missing entries print as 0.
    reason_counts = Dict(:band_exit => 0, :horizon_elapsed => 0,
                         :drawdown => 0, :cash_revisit => 0)
    for r in results, t in r.trigger_log
        t.fired || continue
        reason_counts[t.reason] = get(reason_counts, t.reason, 0) + 1
    end
    summary[interval] = Dict{String,Any}(
        "sharpe_mc"               => sharpe_mc,
        "max_dd_mc"               => max_dd_mc,
        "ann_turnover_mc"         => ann_turnover_mc,
        "ann_return_mc"           => ann_return_mc,
        "n_mpc_triggers_mc"       => n_trig_mc,
        "W_T_over_W0_mc"          => W_T_over_W0_mc,
        "trigger_reason_total"    => Dict(String(k) => v for (k, v) in reason_counts),
        "trigger_reason_per_seed" => Dict(String(k) => v / n_seeds
                                          for (k, v) in reason_counts))

    @printf("interval=%2d days (%d/%d):  Sharpe med=%.3f  IQR=[%.3f,%.3f]  MaxDD%% med=%.1f  trigs/seed med=%d\n",
            interval, j, n_interval,
            median(sharpe_mc), quantile(sharpe_mc, 0.25), quantile(sharpe_mc, 0.75),
            median(max_dd_mc) * 100, Int(round(median(n_trig_mc))))
end
t_elapsed = time() - t_start
@printf("\nWall-clock: %.1f s (%.2f s per interval × %d seeds)\n",
        t_elapsed, t_elapsed / n_interval, n_seeds)

# --- Summary table -----------------------------------------------------------
println("\ncash_revisit_interval sweep summary (across $n_seeds seeds):")
println("-" ^ 118)
@printf("%-9s  %8s %8s %8s   %8s %8s %8s   %8s  %5s %5s %5s %5s\n",
        "interval",
        "Shp_Q25", "Shp_med", "Shp_Q75",
        "DD_min%", "DD_med%", "DD_max%",
        "Turn_med", "band", "dd", "horz", "cash")
println("-" ^ 118)
for interval in CASH_REVISIT_GRID
    s = summary[interval]
    sh, dd, tn = s["sharpe_mc"], s["max_dd_mc"], s["ann_turnover_mc"]
    rps = s["trigger_reason_per_seed"]
    @printf("%9d  %8.3f %8.3f %8.3f   %8.1f %8.1f %8.1f   %8.3f  %5.1f %5.1f %5.1f %5.1f\n",
            interval,
            quantile(sh, 0.25), median(sh), quantile(sh, 0.75),
            minimum(dd) * 100, median(dd) * 100, maximum(dd) * 100,
            median(tn),
            get(rps, "band_exit",       0.0),
            get(rps, "drawdown",        0.0),
            get(rps, "horizon_elapsed", 0.0),
            get(rps, "cash_revisit",    0.0))
end
println("-" ^ 118)

# --- Persist -----------------------------------------------------------------
save_results(joinpath(PATH_OUT, "cash_revisit_sweep.jld2"), Dict(
    "config" => Dict(
        "BACKTEST_MC_SEEDS"     => collect(BACKTEST_MC_SEEDS),
        "n_seeds"               => n_seeds,
        "cash_revisit_grid"     => CASH_REVISIT_GRID,
        "sigma_max_fixed"       => SIGMA_MAX_FIXED,
        "w_max_fixed"           => W_MAX_FIXED,
        "k_turnover_fixed"      => K_TURNOVER_FIXED,
        "hold_out_start"        => string(dates_hold[1]),
        "hold_out_end"          => string(dates_hold[end]),
        "n_days"                => n_days,
        "K"                     => length(basket_tickers),
        "tickers"               => basket_tickers,
        "B_0"                   => B_0,
        "tax_rates"             => Dict("st" => tax_rates.st, "lt" => tax_rates.lt),
        "ewls_half_life_days"   => 252,
        "MPC_spec_fixed"        => Dict("z" => 1.96, "T" => 21,
                                        "N" => 1000, "D_max" => 0.08)),
    "cash_revisit_grid"     => CASH_REVISIT_GRID,
    "summary"               => summary,
    "per_interval_results"  => per_interval_results))
println("\nSaved scripts/data/cash_revisit_sweep.jld2")
