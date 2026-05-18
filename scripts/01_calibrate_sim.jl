# scripts/01_calibrate_sim.jl
# Fit per-ticker SIM on 2014-2024 daily closes; compute ADV; write
# scripts/data/sim_calibration.jld2.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "code"))

using ConstrainedCobbDouglas
using InteractiveUtils
using JLD2
using Statistics

const PATH_INPUTS = joinpath(@__DIR__, "..", "code", "src", "data")
const PATH_OUT = joinpath(@__DIR__, "data")
isdir(PATH_OUT) || mkpath(PATH_OUT)

const SIM_SEED = 2026

println("=" ^ 78)
println("01_calibrate_sim.jl — fitting SIM on 2014-2024")
println("=" ^ 78)
versioninfo()

# Load training-window OHLC
ohlc_train = load_ohlc_jld2(joinpath(PATH_INPUTS,
    "SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2"))
prices_train = ohlc_train.prices
tickers_all  = ohlc_train.tickers
volumes_train = ohlc_train.volumes
n_days, K_all = size(prices_train)
println("Loaded $n_days × $K_all training matrix")

# Load hold-out OHLC for universe filter
ohlc_h1 = load_ohlc_jld2(joinpath(PATH_INPUTS,
    "SP500-Daily-OHLC-1-2-2025-to-12-31-2025.jld2"))
ohlc_h2 = load_ohlc_jld2(joinpath(PATH_INPUTS,
    "SP500-Daily-OHLC-1-2-2026-to-04-22-2026.jld2"))

function full_coverage(prices)
    ok = trues(size(prices, 2))
    for i in axes(prices, 2)
        col = prices[:, i]
        if any(ismissing, col) || any(c -> !isfinite(c) || c <= 0.0, col)
            ok[i] = false
        end
    end
    return ok
end

# load_ohlc_jld2 already drops partial-coverage tickers per file; we still
# need to intersect the three per-file universes by ticker.
h1_set = Set(ohlc_h1.tickers)
h2_set = Set(ohlc_h2.tickers)
keep_ticker = [t in h1_set && t in h2_set for t in tickers_all]
ok_train = full_coverage(prices_train) .& keep_ticker
tickers = tickers_all[ok_train]
prices  = prices_train[:, ok_train]
volumes = volumes_train[:, ok_train]
println("Universe filter: $(length(tickers)) of $K_all tickers survive full coverage")

# Pick market index: first column with ticker "SPY", else equal-weight synthetic
market_idx = findfirst(==("SPY"), tickers)
market_prices = market_idx === nothing ?
    vec(mean(prices; dims = 2)) :
    prices[:, market_idx]

g_m = compute_market_growth(Vector{Float64}(market_prices))
σ_m = std(g_m)
println("σ_m (annualized) = ", round(σ_m; digits = 4))

K = length(tickers)
αs = zeros(K); βs = zeros(K); σ_εs = zeros(K); r²s = zeros(K)
for (i, tk) in enumerate(tickers)
    g_i = compute_market_growth(Vector{Float64}(prices[:, i]))
    n_use = min(length(g_m), length(g_i))
    est = estimate_sim(g_m[1:n_use], g_i[1:n_use], tk)
    αs[i] = est.α; βs[i] = est.β; σ_εs[i] = est.σ_ε; r²s[i] = est.r²
end

adv = Dict(tickers[i] => mean(skipmissing(volumes[:, i])) for i in 1:K)

out = Dict(
    "config" => Dict("SIM_SEED" => SIM_SEED),
    "tickers" => collect(tickers),
    "alpha" => αs, "beta" => βs, "sigma_eps" => σ_εs, "r_squared" => r²s,
    "sigma_market" => σ_m,
    "adv" => adv,
    "n_training_days" => n_days)

outpath = joinpath(PATH_OUT, "sim_calibration.jld2")
save_results(outpath, out)
println("Saved $outpath")
