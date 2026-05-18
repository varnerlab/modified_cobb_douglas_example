# scripts/06_backtest_mc.jl
# 20-seed Monte Carlo backtest of all 6 strategies on the 2025-2026
# hold-out window. Writes scripts/data/backtest_mc_results.jld2.
#
# The two MPC strategies (CDWithMPCStrategy, ConstrainedCDWithMPCStrategy)
# depend on the seed through their `forward_project` MC paths. The 4
# non-MPC strategies are deterministic given prices, so their distributions
# collapse to a single value (we still record them to keep the structure
# uniform).

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

println("=" ^ 78)
println("06_backtest_mc.jl — 20-seed MC backtest of 6 strategies")
println("=" ^ 78)

# Load all artifacts (mirrors scripts/05_backtest_strategies.jl)
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
println("Hold-out: $n_days days, $(length(basket_tickers)) tickers")

# Market prices (SPY or basket EW)
spy_2025_idx = findfirst(==("SPY"), all_tickers_2025)
spy_2026_idx = findfirst(==("SPY"), all_tickers_2026)
if spy_2025_idx !== nothing && spy_2026_idx !== nothing
    market_prices = vcat(Vector{Float64}(ohlc_2025.prices[:, spy_2025_idx]),
                         Vector{Float64}(ohlc_2026.prices[:, spy_2026_idx]))
else
    market_prices = vec(mean(prices_hold; dims = 2))
end

# SIM params slice for basket
sim_calib_tickers = sim_calib["tickers"]
sim_col = Dict(t => i for (i, t) in enumerate(sim_calib_tickers))
αs = Float64[sim_calib["alpha"][sim_col[t]] for t in basket_tickers]
βs = Float64[sim_calib["beta"][sim_col[t]] for t in basket_tickers]
σ_εs = Float64[sim_calib["sigma_eps"][sim_col[t]] for t in basket_tickers]
σ_m = Float64(sim_calib["sigma_market"])

# EWLS init from frozen 2014-2024 OLS estimates
sim_init = Dict(basket_tickers[i] => ewls_init(αs[i], βs[i], σ_εs[i];
    half_life = 252.0, prior_weight = 252.0) for i in eachindex(basket_tickers))

# Load pretrained JumpHMM market surrogate
market_model = load(joinpath(PATH_INPUTS, "pretrained-jumphmm-market-surrogate.jld2"),
    "model")

# ADV from sim_calib
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

# The six strategies — identical to scripts/05_backtest_strategies.jl
spec = MyMPCSpec(z = 1.96, T = 21, N = 1000, D_max = 0.08)
# σ_max in growth-rate-vol units; matches script 05.
strategies = MyAllocationStrategy[
    EqualWeightStrategy(),
    MinVarBuyHoldStrategy(),
    UnconstrainedCDStrategy(),
    CostAwareMVStrategy(κ = 5.0, c = 0.0005),
    CDWithMPCStrategy(spec = spec),
    ConstrainedCDWithMPCStrategy(spec = spec,
        σ_max = 2.54, K_turnover = 0.10 * B_0, w_max = 0.20)]

n_seeds = length(BACKTEST_MC_SEEDS)
println("\nRunning $(length(strategies)) strategies × $n_seeds seeds = ",
        length(strategies) * n_seeds, " backtests")
println()

per_seed_results = Vector{Dict{String,MyBacktestResult}}(undef, n_seeds)

t_start = time()
for (i, s) in enumerate(BACKTEST_MC_SEEDS)
    res = compare_strategies(strategies, env, cost_model, tax_rates;
                             B₀ = B_0, rng_seed = s)
    per_seed_results[i] = res
    ccd_mpc = res["ConstrainedCDWithMPCStrategy"]
    @printf("Seed %d (%d/%d):  CCD-MPC Sharpe=%.3f, MaxDD%%=%.1f\n",
            s, i, n_seeds,
            ccd_mpc.summary.ann_sharpe,
            ccd_mpc.summary.max_drawdown * 100)
end
t_elapsed = time() - t_start
println()
@printf("Wall-clock for %d seeds: %.1f s (%.2f s/seed)\n",
        n_seeds, t_elapsed, t_elapsed / n_seeds)

# Build per-strategy aggregated vectors of length n_seeds
strat_names = sort(collect(keys(per_seed_results[1])))
summary = Dict{String,Dict{String,Any}}()
for name in strat_names
    sharpe_mc        = Float64[per_seed_results[i][name].summary.ann_sharpe       for i in 1:n_seeds]
    max_dd_mc        = Float64[per_seed_results[i][name].summary.max_drawdown     for i in 1:n_seeds]
    ann_turnover_mc  = Float64[per_seed_results[i][name].summary.ann_turnover     for i in 1:n_seeds]
    ann_return_mc    = Float64[per_seed_results[i][name].summary.ann_return       for i in 1:n_seeds]
    n_mpc_triggers_mc = Int[per_seed_results[i][name].summary.n_mpc_triggers     for i in 1:n_seeds]
    W_T_over_W0_mc   = Float64[per_seed_results[i][name].wealth_after_cost_pretax[end] /
                               per_seed_results[i][name].wealth_after_cost_pretax[1]
                               for i in 1:n_seeds]
    summary[name] = Dict{String,Any}(
        "sharpe_mc"          => sharpe_mc,
        "max_dd_mc"          => max_dd_mc,
        "W_T_over_W0_mc"     => W_T_over_W0_mc,
        "ann_turnover_mc"    => ann_turnover_mc,
        "n_mpc_triggers_mc"  => n_mpc_triggers_mc,
        "ann_return_mc"      => ann_return_mc)
end

# Headline quantile table
println("\nDistribution across $n_seeds seeds (Sharpe / MaxDD% / W_T/W_0):")
println("-" ^ 110)
@printf("%-35s  %8s %8s %8s %8s %8s   %8s %8s %8s   %8s %8s %8s\n",
        "Strategy",
        "Shp_min", "Shp_Q25", "Shp_med", "Shp_Q75", "Shp_max",
        "DD_min%", "DD_med%", "DD_max%",
        "WT_min", "WT_med", "WT_max")
println("-" ^ 110)
# Sort strategies by median Sharpe descending for readability
ordered_names = sort(strat_names; by = n -> -median(summary[n]["sharpe_mc"]))
for name in ordered_names
    sh = summary[name]["sharpe_mc"]
    dd = summary[name]["max_dd_mc"]
    wt = summary[name]["W_T_over_W0_mc"]
    @printf("%-35s  %8.3f %8.3f %8.3f %8.3f %8.3f   %8.1f %8.1f %8.1f   %8.3f %8.3f %8.3f\n",
            name,
            minimum(sh), quantile(sh, 0.25), median(sh), quantile(sh, 0.75), maximum(sh),
            minimum(dd) * 100, median(dd) * 100, maximum(dd) * 100,
            minimum(wt), median(wt), maximum(wt))
end
println("-" ^ 110)

# Persist
save_results(joinpath(PATH_OUT, "backtest_mc_results.jld2"), Dict(
    "config" => Dict(
        "BACKTEST_MC_SEEDS" => collect(BACKTEST_MC_SEEDS),
        "n_seeds" => length(BACKTEST_MC_SEEDS),
        "hold_out_start" => string(dates_hold[1]),
        "hold_out_end" => string(dates_hold[end]),
        "n_days" => n_days,
        "K" => length(basket_tickers),
        "tickers" => basket_tickers,
        "B_0" => 100_000.0,
        "tax_rates" => Dict("st" => tax_rates.st, "lt" => tax_rates.lt),
        "ewls_half_life_days" => 252,
        "sigma_max" => 2.54,
        "K_turnover" => 0.10 * 100_000.0,
        "w_max" => 0.20),
    "summary" => summary,
    "per_seed_results" => per_seed_results))
println("\nSaved scripts/data/backtest_mc_results.jld2")
