# scripts/03_train_bandit_mc.jl
# 30-seed Monte Carlo. Reuses the per-sector bandit machinery from 02.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "code"))
using ConstrainedCobbDouglas
using JLD2
using Random
using Statistics

const PATH_INPUTS = joinpath(@__DIR__, "..", "code", "src", "data")
const PATH_OUT = joinpath(@__DIR__, "data")
const BANDIT_MC_SEEDS = 1001:1030
const K_BASKET = 33
const ITERS_PER_ARM = 50
const ITERS_MAX = 5000
const ITERS_MIN = 500
const FORWARD_HORIZON = 21
const TRAIN_OFFSET = 252

println("=" ^ 78)
println("03_train_bandit_mc.jl — 30 seeds")
println("=" ^ 78)

sim_calib = load_results(joinpath(PATH_OUT, "sim_calibration.jld2"))
ohlc_train = load_ohlc_jld2(joinpath(PATH_INPUTS,
    "SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2"))
tickers = sim_calib["tickers"]
tickers_full = ohlc_train.tickers
col_of = Dict(t => i for (i, t) in enumerate(tickers_full))
keep_cols = [col_of[t] for t in tickers]
price_matrix = Matrix{Float64}(ohlc_train.prices[:, keep_cols])
market_idx = findfirst(==("SPY"), tickers)
market_prices = market_idx === nothing ? vec(mean(price_matrix; dims = 2)) : price_matrix[:, market_idx]
g_m = compute_market_growth(Vector{Float64}(market_prices))
short_ema = compute_ema(Vector{Float64}(market_prices); window = 21)
long_ema  = compute_ema(Vector{Float64}(market_prices); window = 63)
λ_series  = compute_lambda(short_ema, long_ema; G = 50.0)
gm_series = vcat([0.0], g_m)
n_days = length(market_prices)
sector_of, _ = load_sector_map(tickers, joinpath(PATH_INPUTS, "sp500-sectors.csv"))
sector_groups = Dict{String,Vector{Int}}()
for (i, t) in enumerate(tickers)
    haskey(sector_of, t) && push!(get!(sector_groups, sector_of[t], Int[]), i)
end
quotas = assign_quotas(sector_groups, K_BASKET)
sim_params = Dict(tickers[i] => (Float64(sim_calib["alpha"][i]),
                                 Float64(sim_calib["beta"][i]),
                                 Float64(sim_calib["sigma_eps"][i]))
                  for i in eachindex(tickers))
train_offset = TRAIN_OFFSET
train_last = n_days - FORWARD_HORIZON - 1

per_seed_tickers = Vector{Vector{String}}()
per_seed_indices = Vector{Vector{Int}}()
per_seed_best_means = Vector{Dict{String,Float64}}()
for (k, seed) in enumerate(BANDIT_MC_SEEDS)
    println("Seed $seed ($(k)/$(length(BANDIT_MC_SEEDS)))")
    rng_master = MersenneTwister(seed)
    sector_results = Dict{String,NamedTuple}()
    for s in sort(collect(keys(sector_groups)))
        sec_idx = sector_groups[s]
        q = quotas[s]
        n_arms = binomial(length(sec_idx), q)
        iters = clamp(n_arms * ITERS_PER_ARM, ITERS_MIN, ITERS_MAX)
        sub_seed = rand(rng_master, 1:10^9)
        res = train_sector_bandit(sec_idx, q, train_offset, train_last,
                FORWARD_HORIZON, price_matrix, sim_params,
                Vector{String}(tickers), gm_series, λ_series;
                iters = iters, seed = sub_seed)
        sector_results[s] = res
    end
    indices = Int[]
    for s in sort(collect(keys(sector_groups)))
        append!(indices, sector_results[s].best_arm)
    end
    push!(per_seed_indices, indices)
    push!(per_seed_tickers, tickers[indices])
    push!(per_seed_best_means,
          Dict(s => sector_results[s].best_mean for s in keys(sector_results)))
end

save_results(joinpath(PATH_OUT, "per_sector_bandit_mc_results.jld2"), Dict(
    "config" => Dict("BANDIT_MC_SEEDS" => collect(BANDIT_MC_SEEDS),
                     "K_BASKET" => K_BASKET, "FORWARD_HORIZON" => FORWARD_HORIZON),
    "quotas" => quotas,
    "per_seed_tickers" => per_seed_tickers,
    "per_seed_indices" => per_seed_indices,
    "per_seed_best_means" => per_seed_best_means))
println("Saved scripts/data/per_sector_bandit_mc_results.jld2")
