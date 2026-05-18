# scripts/02_train_bandit.jl
# Single-seed run of the per-sector bandit. Dev sanity check before MC.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "code"))
using ConstrainedCobbDouglas
using JLD2
using Random
using Statistics

const PATH_INPUTS = joinpath(@__DIR__, "..", "code", "src", "data")
const PATH_OUT = joinpath(@__DIR__, "data")

const BANDIT_SEED = 2026
const K_BASKET = 22
const ITERS_PER_ARM = 50
const ITERS_MAX = 5000
const ITERS_MIN = 500
const FORWARD_HORIZON = 21
const TRAIN_OFFSET = 252
println("=" ^ 78)
println("02_train_bandit.jl — single seed = $BANDIT_SEED")
println("=" ^ 78)

sim_calib = load_results(joinpath(PATH_OUT, "sim_calibration.jld2"))
ohlc_train = load_ohlc_jld2(joinpath(PATH_INPUTS,
    "SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2"))
tickers = sim_calib["tickers"]
prices_full = ohlc_train.prices
tickers_full = ohlc_train.tickers
col_of = Dict(t => i for (i, t) in enumerate(tickers_full))
keep_cols = [col_of[t] for t in tickers]
price_matrix = Matrix{Float64}(prices_full[:, keep_cols])

market_idx = findfirst(==("SPY"), tickers)
market_prices = market_idx === nothing ? vec(mean(price_matrix; dims = 2)) : price_matrix[:, market_idx]

g_m = compute_market_growth(Vector{Float64}(market_prices))
short_ema = compute_ema(Vector{Float64}(market_prices); window = 21)
long_ema  = compute_ema(Vector{Float64}(market_prices); window = 63)
λ_series  = compute_lambda(short_ema, long_ema)
n_days = length(market_prices)
gm_series = vcat([0.0], g_m)

sector_csv = joinpath(PATH_INPUTS, "sp500-sectors.csv")
sector_of, dropped = load_sector_map(tickers, sector_csv)
sector_groups = Dict{String,Vector{Int}}()
for (i, t) in enumerate(tickers)
    if haskey(sector_of, t)
        push!(get!(sector_groups, sector_of[t], Int[]), i)
    end
end
println("Sectors: $(length(sector_groups));  dropped: $(length(dropped))")

quotas = assign_quotas(sector_groups, K_BASKET)
println("Quotas:")
for s in keys(sector_groups)
    println("  ", rpad(s, 25), "  N_s = ", lpad(length(sector_groups[s]), 3),
        "   q_s = ", quotas[s])
end

sim_params = Dict(tickers[i] => (Float64(sim_calib["alpha"][i]),
                                 Float64(sim_calib["beta"][i]),
                                 Float64(sim_calib["sigma_eps"][i]))
                  for i in eachindex(tickers))

train_offset = TRAIN_OFFSET
train_last   = n_days - FORWARD_HORIZON - 1

rng_master = MersenneTwister(BANDIT_SEED)
sector_results = Dict{String,NamedTuple}()
for s in sort(collect(keys(sector_groups)))
    sec_idx = sector_groups[s]
    q = quotas[s]
    n_arms = binomial(length(sec_idx), q)
    iters = clamp(n_arms * ITERS_PER_ARM, ITERS_MIN, ITERS_MAX)
    seed = rand(rng_master, 1:10^9)
    t0 = time()
    res = train_sector_bandit(sec_idx, q, train_offset, train_last,
            FORWARD_HORIZON, price_matrix, sim_params,
            Vector{String}(tickers), gm_series, λ_series;
            iters = iters, seed = seed)
    sector_results[s] = res
    println("  ", rpad(s, 25),
        "  iters=", lpad(iters, 5),
        "  best_mean=", round(res.best_mean; digits = 4),
        "  ", round(time() - t0; digits = 1), "s")
end

basket_indices = Int[]
for s in sort(collect(keys(sector_groups)))
    append!(basket_indices, sector_results[s].best_arm)
end
basket_tickers = tickers[basket_indices]
println("\nAssembled basket ($(length(basket_tickers)) tickers): ",
        join(basket_tickers, ", "))

out = Dict(
    "config" => Dict("BANDIT_SEED" => BANDIT_SEED, "K_BASKET" => K_BASKET,
                     "FORWARD_HORIZON" => FORWARD_HORIZON),
    "quotas" => quotas,
    "sector_best_arms" => Dict(s => sector_results[s].best_arm for s in keys(sector_results)),
    "sector_best_means" => Dict(s => sector_results[s].best_mean for s in keys(sector_results)),
    "basket_tickers" => collect(basket_tickers),
    "basket_indices" => collect(basket_indices))
save_results(joinpath(PATH_OUT, "per_sector_bandit_results.jld2"), out)
println("Saved scripts/data/per_sector_bandit_results.jld2")
