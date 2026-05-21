# scripts/lambda_g_sweep.jl
#
# Diagnostic sweep over the gain hyperparameter G in the signed
# regime-lens compute_lambda(short_ema, long_ema; G).
#
# For each candidate G we:
#   1. Compute the λ time series on real SPY prices (2014–2024 training window).
#   2. Report distributional stats (min, median, max, std, fraction bearish).
#   3. Compute γ across the frozen basket at every trading day, using the
#      EWLS-initialized (α, β) and per-day smoothed market growth gm_t.
#   4. Report γ-side stats: any NaN/Inf? Range of γ? Fraction of names with
#      γ ≤ 0 (the "non-preferred / cash trigger" share).
#
# Output: pretty-printed sweep table; nothing saved to disk (diagnostic only).

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "code"))

using ConstrainedCobbDouglas
using JLD2
using Statistics
using Printf
using DataFrames
using PrettyTables

const PATH_INPUTS = joinpath(@__DIR__, "..", "code", "src", "data")
const PATH_OUT    = joinpath(@__DIR__, "data")

# --- Load data ---------------------------------------------------------------

println("=" ^ 78)
println("lambda_g_sweep.jl — G sweep on signed compute_lambda")
println("=" ^ 78)

sim_calib    = load_results(joinpath(PATH_OUT, "sim_calibration.jld2"))
frozen_basket = load_results(joinpath(PATH_OUT, "frozen_basket.jld2"))
ohlc_train   = load_ohlc_jld2(joinpath(PATH_INPUTS,
    "SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2"))

basket_tickers = frozen_basket["tickers"]
tickers_full   = ohlc_train.tickers
prices_full    = ohlc_train.prices
col_of = Dict(t => i for (i, t) in enumerate(tickers_full))

market_idx = findfirst(==("SPY"), tickers_full)
@assert market_idx !== nothing "SPY not in OHLC universe"
market_prices = Vector{Float64}(prices_full[:, market_idx])

short_ema = compute_ema(market_prices; window = 21)
long_ema  = compute_ema(market_prices; window = 63)
gm_full   = compute_market_growth(market_prices)
gm_series = vcat([0.0], gm_full)   # align with EMA length

# SIM parameter dict for compute_preference_weights — same shape as the runtime
# expects: Dict{String,Tuple{Float64,Float64,Float64}} = (α, β, σ_ε)
sim_tickers = sim_calib["tickers"]
sim_idx = Dict(t => i for (i, t) in enumerate(sim_tickers))
sim_params_all = Dict{String,Tuple{Float64,Float64,Float64}}()
for t in basket_tickers
    i = sim_idx[t]
    sim_params_all[t] = (sim_calib["alpha"][i], sim_calib["beta"][i], sim_calib["sigma_eps"][i])
end

# Drop the EMA warmup region (first 63 days) so the sweep reflects the operating
# regime rather than initialization transient.
const WARMUP = 63
days = WARMUP+1:length(market_prices)

println("Basket size:           ", length(basket_tickers))
println("Market price series:   ", length(market_prices), " trading days")
println("Days after warmup:     ", length(days))
println()

# --- Sweep ------------------------------------------------------------------

G_grid = [0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

K = length(basket_tickers)

# Pre-compute the G=0 baseline (λ = 0 every day) so each row can report the
# fraction of (day, ticker) γ-signs that *flip* relative to the regime-lens-off
# allocator — the decision-relevance proxy.
γ_baseline = Matrix{Float64}(undef, length(days), K)
for (j, t) in enumerate(days)
    γ_baseline[j, :] = compute_preference_weights(sim_params_all, basket_tickers,
                                                  gm_series[t], 0.0)
end
sign_baseline = sign.(γ_baseline)

rows = NamedTuple[]
for G in G_grid
    λ_series = compute_lambda(short_ema, long_ema; G = G)
    λ_op = λ_series[days]

    λ_min   = minimum(λ_op)
    λ_max   = maximum(λ_op)
    λ_med   = median(λ_op)
    λ_std   = std(λ_op)
    pct_bear = 100.0 * count(>(0.0), λ_op) / length(λ_op)

    γ_any_nan = false
    γ_any_inf = false
    γ_min_seen = Inf
    γ_max_seen = -Inf
    n_nonpref = 0
    n_total   = 0
    n_flip    = 0
    abs_delta_sum = 0.0   # Σ|γ_G - γ_0| over all (day, ticker) — magnitude shift vs baseline
    pref_per_day = Vector{Int}(undef, length(days))
    for (j, t) in enumerate(days)
        γ = compute_preference_weights(sim_params_all, basket_tickers,
                                       gm_series[t], λ_series[t])
        npref_today = 0
        for (k, v) in enumerate(γ)
            n_total += 1
            if isnan(v); γ_any_nan = true; continue; end
            if isinf(v); γ_any_inf = true; continue; end
            γ_min_seen = min(γ_min_seen, v)
            γ_max_seen = max(γ_max_seen, v)
            if v ≤ 0.0
                n_nonpref += 1
            else
                npref_today += 1
            end
            if sign(v) != sign_baseline[j, k]
                n_flip += 1
            end
            abs_delta_sum += abs(v - γ_baseline[j, k])
        end
        pref_per_day[j] = npref_today
    end
    pct_nonpref = 100.0 * n_nonpref / n_total
    pct_flip    = 100.0 * n_flip / n_total
    mean_pref   = mean(pref_per_day)
    mean_abs_delta = abs_delta_sum / n_total

    push!(rows, (
        G          = G,
        λ_min      = λ_min,
        λ_med      = λ_med,
        λ_max      = λ_max,
        pct_bear   = pct_bear,
        γ_min      = γ_min_seen,
        γ_max      = γ_max_seen,
        pct_γ_le0  = pct_nonpref,
        mean_pref  = mean_pref,
        pct_flip   = pct_flip,
        mean_abs_delta = mean_abs_delta,
        nan_or_inf = γ_any_nan || γ_any_inf,
    ))
end

df = DataFrame(rows)
# Pre-format numeric columns to strings so we don't depend on PrettyTables
# formatter API (which moves between major versions).
df_disp = DataFrame(
    "G"            => [@sprintf("%.1f", r.G)            for r in eachrow(df)],
    "λ min"        => [@sprintf("%.3f", r.λ_min)        for r in eachrow(df)],
    "λ med"        => [@sprintf("%.3f", r.λ_med)        for r in eachrow(df)],
    "λ max"        => [@sprintf("%.3f", r.λ_max)        for r in eachrow(df)],
    "% bear"       => [@sprintf("%.1f%%", r.pct_bear)   for r in eachrow(df)],
    "γ min"        => [@sprintf("%.3f", r.γ_min)        for r in eachrow(df)],
    "γ max"        => [@sprintf("%.3f", r.γ_max)        for r in eachrow(df)],
    "% γ≤0"        => [@sprintf("%.1f%%", r.pct_γ_le0)  for r in eachrow(df)],
    "mean pref/K"  => [@sprintf("%.1f / %d", r.mean_pref, K) for r in eachrow(df)],
    "% γ sign flip" => [@sprintf("%.1f%%", r.pct_flip)  for r in eachrow(df)],
    "mean |Δγ vs G=0|" => [@sprintf("%.4f", r.mean_abs_delta) for r in eachrow(df)],
    "NaN/Inf?"     => [string(r.nan_or_inf)             for r in eachrow(df)],
)
pretty_table(df_disp;
    table_format = TextTableFormat(borders = text_table_borders__compact),
    fit_table_in_display_horizontally = false,
    fit_table_in_display_vertically   = false,
)

println()
println("Notes:")
println("  - λ min/med/max are over post-warmup operating days only.")
println("  - % bear = fraction of days with λ > 0 (short EMA < long EMA).")
println("  - γ range and % γ≤0 are over (day × ticker) pairs in the basket.")
println("  - mean pref/K = average number of preferred names per day (γ > 0),")
println("    out of K basket size. Lower means fewer names to allocate over.")
println("  - % γ sign flip = fraction of (day, ticker) pairs where the γ sign")
println("    differs from the G=0 (lens-off) baseline. Sign-invariance of γ")
println("    under λ is structural (see lambda_swap_note.md); 0% is expected.")
println("  - mean |Δγ vs G=0| = average magnitude shift of γ versus the lens-")
println("    off baseline. This is where the lens actually does its work — see")
println("    the β-bucket breakdown below for the directional tilt.")
println("  - NaN/Inf? = true means the γ formula overflowed for at least one")
println("    (day, ticker) — a downstream stability red flag.")

# --- β-bucket × regime breakdown ---------------------------------------------
#
# The G sweep above shows γ sign never flips with λ, so the lens does not gate
# which names are preferred. The tilt happens inside the preferred set, via the
# magnitude prefactor |β_i|^(1-λ). To see that empirically, we split the basket
# by β tercile and split days by λ sign, then report mean γ and mean allocator
# weight (γ_i / Σγ+) for preferred names per (β-bucket × regime) cell.
#
# Expected pattern if the lens works:
#   bullish (λ<0):  high-β bucket has the largest mean weight (1 - λ > 1, so
#                   |β|^(1-λ) amplifies high-β names);
#   bearish (λ>1):  low-β bucket has the largest mean weight (1 - λ < 0, so
#                   |β|^(1-λ) amplifies low-β names — the |β|<1 ones).

println()
println("=" ^ 78)
println("β-bucket × regime breakdown — does the lens tilt composition?")
println("=" ^ 78)

basket_betas = Float64[sim_params_all[t][2] for t in basket_tickers]
β_lo_thresh = quantile(abs.(basket_betas), 1/3)
β_hi_thresh = quantile(abs.(basket_betas), 2/3)
bucket_of(β) = abs(β) ≤ β_lo_thresh ? :low :
               abs(β) ≥ β_hi_thresh ? :high : :mid
ticker_bucket = Symbol[bucket_of(b) for b in basket_betas]
println(@sprintf("β tercile cutoffs (|β|): low ≤ %.2f, high ≥ %.2f", β_lo_thresh, β_hi_thresh))
n_per_bucket = Dict(b => count(==(b), ticker_bucket) for b in (:low, :mid, :high))
println("Tickers per bucket: low=$(n_per_bucket[:low]), mid=$(n_per_bucket[:mid]), high=$(n_per_bucket[:high])")
println()

G_for_breakdown = [1.0, 20.0, 100.0]
bucket_rows = NamedTuple[]
for G in G_for_breakdown
    λ_series = compute_lambda(short_ema, long_ema; G = G)
    # accumulators: (bucket, regime) => Vector{Float64} of normalized weights
    weight_accum = Dict{Tuple{Symbol,Symbol},Vector{Float64}}()
    γ_accum      = Dict{Tuple{Symbol,Symbol},Vector{Float64}}()
    for key in Iterators.product((:low, :mid, :high), (:bullish, :bearish))
        weight_accum[key] = Float64[]
        γ_accum[key]      = Float64[]
    end
    for t in days
        λ_t = λ_series[t]
        regime = λ_t < 0.0 ? :bullish : (λ_t > 0.0 ? :bearish : :neutral)
        regime == :neutral && continue   # ignore the (rare) λ_t == 0 days
        γ = compute_preference_weights(sim_params_all, basket_tickers,
                                       gm_series[t], λ_t)
        γ_pos = max.(γ, 0.0)
        s = sum(γ_pos)
        s == 0.0 && continue              # full cash day; no preferred mass to split
        weights = γ_pos ./ s
        for (k, v) in enumerate(γ)
            v ≤ 0.0 && continue           # only preferred names contribute to tilt
            b = ticker_bucket[k]
            push!(weight_accum[(b, regime)], weights[k])
            push!(γ_accum[(b, regime)], v)
        end
    end
    for regime in (:bullish, :bearish)
        for bucket in (:low, :mid, :high)
            ws = weight_accum[(bucket, regime)]
            gs = γ_accum[(bucket, regime)]
            push!(bucket_rows, (
                G        = G,
                regime   = regime,
                bucket   = bucket,
                n        = length(ws),
                mean_γ   = isempty(gs) ? NaN : mean(gs),
                mean_w   = isempty(ws) ? NaN : mean(ws),
                share_w  = isempty(ws) ? NaN : sum(ws),  # raw weight mass in this bucket-day
            ))
        end
    end
end

# Display: mean preferred γ and mean allocator weight per (G × regime × bucket)
df_bucket = DataFrame(bucket_rows)
df_bucket_disp = DataFrame(
    "G"           => [@sprintf("%.0f", r.G)        for r in eachrow(df_bucket)],
    "regime"      => [string(r.regime)             for r in eachrow(df_bucket)],
    "β bucket"    => [string(r.bucket)             for r in eachrow(df_bucket)],
    "n (pref)"    => [string(r.n)                  for r in eachrow(df_bucket)],
    "mean γ"      => [@sprintf("%.4f", r.mean_γ)   for r in eachrow(df_bucket)],
    "mean weight" => [@sprintf("%.4f", r.mean_w)   for r in eachrow(df_bucket)],
)
pretty_table(df_bucket_disp;
    table_format = TextTableFormat(borders = text_table_borders__compact),
    fit_table_in_display_horizontally = false,
    fit_table_in_display_vertically   = false,
)

println()
println("Read this table by row pair: at each G, compare 'mean weight' across")
println("β buckets within a single regime.")
println("  bullish: if the lens tilts toward risk, mean weight for high-β > low-β.")
println("  bearish: if the lens tilts defensive, mean weight for low-β > high-β.")
println("'mean weight' is the average allocator weight γ_i / Σγ⁺ for preferred")
println("(day, ticker) pairs in that cell; absolute level depends on bucket size,")
println("so compare directionally between cells with similar n.")
