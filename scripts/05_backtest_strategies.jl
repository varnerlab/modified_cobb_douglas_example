# scripts/05_backtest_strategies.jl
# Walk all 6 strategies on the 2025-2026 hold-out window. Write
# scripts/data/backtest_results.jld2.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "code"))

using ConstrainedCobbDouglas
using JLD2
using JumpHMM
using Statistics
using Dates

const PATH_INPUTS = joinpath(@__DIR__, "..", "code", "src", "data")
const PATH_OUT    = joinpath(@__DIR__, "data")
const BACKTEST_RNG_SEED = 2026
const B_0 = 100_000.0

println("=" ^ 78)
println("05_backtest_strategies.jl — 6-strategy bake-off")
println("=" ^ 78)

# Load all artifacts
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

# Load pretrained JumpHMM market surrogate.
# The file stores the model under the key "model" (other keys are metadata:
# "nu", "ticker", "n_states", "training_date_range", "dt", "n_training_days",
# "rf").
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

# The six strategies
spec = MyMPCSpec(z = 1.96, T = 21, N = 1000, D_max = 0.08)
# σ_max is in growth-rate-vol units (annualized). EW basket σ ≈ 2.99 in
# these units (σ_m ≈ 2.74), so σ_max ≈ 0.85·EW_σ ≈ 2.54 keeps the
# covariance cap binding on most strategy days without trivial infeasibility.
strategies = MyAllocationStrategy[
    EqualWeightStrategy(),
    MinVarBuyHoldStrategy(),
    UnconstrainedCDStrategy(),
    CostAwareMVStrategy(κ = 5.0, c = 0.0005),
    CDWithMPCStrategy(spec = spec),
    ConstrainedCDWithMPCStrategy(spec = spec,
        σ_max = 2.54, K_turnover = 0.10 * B_0, w_max = 0.20)]

println("\nRunning $(length(strategies)) strategies:")
results = compare_strategies(strategies, env, cost_model, tax_rates;
    B₀ = B_0, rng_seed = BACKTEST_RNG_SEED)

println("\nHeadline metrics (after-cost, after-tax):")
println(rpad("Strategy", 35), "  ", rpad("Sharpe", 10),
        rpad("MaxDD%", 10), rpad("Turnover", 10), "n_trig")
for (name, r) in results
    sm = r.summary
    println(rpad(name, 35), "  ",
        rpad(round(sm.ann_sharpe; digits = 3), 10),
        rpad(round(sm.max_drawdown * 100; digits = 1), 10),
        rpad(round(sm.ann_turnover; digits = 3), 10),
        sm.n_mpc_triggers)
end

# Persist
out = Dict(
    "config" => Dict(
        "hold_out_start" => string(dates_hold[1]),
        "hold_out_end" => string(dates_hold[end]),
        "n_days" => n_days,
        "K" => length(basket_tickers),
        "tickers" => basket_tickers,
        "B_0" => B_0,
        "rng_seed" => BACKTEST_RNG_SEED,
        "tax_rates" => Dict("st" => tax_rates.st, "lt" => tax_rates.lt),
        "ewls_half_life_days" => 252),
    "dates"   => dates_hold,
    "results" => results)
save_results(joinpath(PATH_OUT, "backtest_results.jld2"), out)
println("\nSaved scripts/data/backtest_results.jld2")
