# scripts/07b_sigma_max_fine_sweep.jl
#
# Fine-grained σ_max sweep below the 07 grid floor of 1.5, to fill the gap
# in [0.05, 1.5] so we can see how the live trial's σ_max = 0.12 actually
# performs under corrected Σ units. Mirrors 07's environment and constraints
# (K_turnover_frac × B_0, w_max = 0.20, 20-seed MC, frozen basket); only the
# σ_max grid differs.
#
# Writes scripts/data/sigma_max_fine_sweep.jld2.

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
const SIGMA_MAX_GRID = [0.05, 0.10, 0.12, 0.20, 0.30, 0.50, 1.00, 1.50]

println("=" ^ 78)
println("07b_sigma_max_fine_sweep.jl — fine σ_max sweep, ConstrainedCDWithMPC")
println("=" ^ 78)

# --- Mirror 07 environment setup ---------------------------------------------
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
        length(SIGMA_MAX_GRID), " σ_max values, ",
        length(BACKTEST_MC_SEEDS), " seeds")

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

n_seeds  = length(BACKTEST_MC_SEEDS)
n_sigmas = length(SIGMA_MAX_GRID)
println("\nRunning ConstrainedCDWithMPC at $n_sigmas σ_max × $n_seeds seeds = ",
        n_sigmas * n_seeds, " backtests\n")

per_sigma_results = Dict{Float64,Vector{MyBacktestResult}}()
summary = Dict{Float64,Dict{String,Any}}()

t_start = time()
for (j, σmax) in enumerate(SIGMA_MAX_GRID)
    strat = ConstrainedCDWithMPCStrategy(spec = spec,
        σ_max = σmax, K_turnover = 0.10 * B_0, w_max = 0.20)

    results = Vector{MyBacktestResult}(undef, n_seeds)
    for (i, s) in enumerate(BACKTEST_MC_SEEDS)
        results[i] = run_backtest(strat, env, cost_model, tax_rates;
                                  B₀ = B_0, rng_seed = s)
    end
    per_sigma_results[σmax] = results

    sharpe_mc       = Float64[r.summary.ann_sharpe       for r in results]
    max_dd_mc       = Float64[r.summary.max_drawdown     for r in results]
    ann_turnover_mc = Float64[r.summary.ann_turnover     for r in results]
    ann_return_mc   = Float64[r.summary.ann_return       for r in results]
    n_trig_mc       = Int[r.summary.n_mpc_triggers       for r in results]
    W_T_over_W0_mc  = Float64[r.wealth_after_cost_pretax[end] /
                              r.wealth_after_cost_pretax[1] for r in results]

    summary[σmax] = Dict(
        "sigma_max"           => σmax,
        "ann_sharpe_med"      => median(sharpe_mc),
        "max_drawdown_med"    => median(max_dd_mc),
        "ann_turnover_med"    => median(ann_turnover_mc),
        "ann_return_med"      => median(ann_return_mc),
        "n_mpc_triggers_med"  => median(n_trig_mc),
        "W_T_over_W0_med"     => median(W_T_over_W0_mc),
    )

    @printf("σ_max=%6.3f (%d/%d):  Sharpe med=%.3f  MaxDD%% med=%.1f  trigs/seed med=%.0f  W_T/W_0 med=%.3f\n",
        σmax, j, n_sigmas,
        median(sharpe_mc),
        100 * median(max_dd_mc),
        median(n_trig_mc),
        median(W_T_over_W0_mc),
    )
end
t_elapsed = time() - t_start
println("\nElapsed: $(round(t_elapsed; digits=1))s")

save_results(joinpath(PATH_OUT, "sigma_max_fine_sweep.jld2"),
    Dict("per_sigma_results" => per_sigma_results,
         "summary"           => summary,
         "sigma_max_grid"    => SIGMA_MAX_GRID,
         "seeds"             => collect(BACKTEST_MC_SEEDS),
         "B_0"               => B_0,
         "w_max"             => 0.20,
         "K_turnover"        => 0.10 * B_0,
         "spec"              => Dict("z" => 1.96, "T" => 21, "N" => 1000,
                                      "D_max" => 0.08),
         "trial_sigma_max"   => 0.12,
    ))
println("\nSaved scripts/data/sigma_max_fine_sweep.jld2")
