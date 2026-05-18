# scripts/04_select_basket.jl
# Pick the median-Sharpe seed from the MC results and write the frozen basket.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "code"))
using ConstrainedCobbDouglas
using JLD2
using Statistics

const PATH_OUT = joinpath(@__DIR__, "data")

mc = load_results(joinpath(PATH_OUT, "per_sector_bandit_mc_results.jld2"))
seeds = mc["config"]["BANDIT_MC_SEEDS"]
per_seed_means = mc["per_seed_best_means"]
scores = [mean(values(d)) for d in per_seed_means]
order = sortperm(scores)
median_idx = order[ceil(Int, length(order) / 2)]
println("Median-score seed: ", seeds[median_idx], " (score = ", round(scores[median_idx]; digits = 4), ")")
frozen_tickers = mc["per_seed_tickers"][median_idx]
println("Basket: ", join(frozen_tickers, ", "))

out = Dict(
    "tickers"       => collect(frozen_tickers),
    "seed_id"       => seeds[median_idx],
    "sector_quotas" => mc["quotas"],
    "mc_summary"    => Dict(
        "scores_min"    => minimum(scores),
        "scores_median" => median(scores),
        "scores_max"    => maximum(scores),
        "n_seeds"       => length(seeds)))
save_results(joinpath(PATH_OUT, "frozen_basket.jld2"), out)
println("Saved scripts/data/frozen_basket.jld2")
