# scripts/lambda_swap_deltas.jl
#
# Diagnostic: compare the post-G=50 backtest + sweep artifacts against the
# .pre_lambda_swap.jld2 backups captured immediately before the lambda swap.
# Prints headline bake-off deltas (median Sharpe / MaxDD / W_T) and the
# top-line summary deltas for each of the four sweeps.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "code"))

using ConstrainedCobbDouglas
using JLD2
using Statistics
using Printf

const PATH_OUT = joinpath(@__DIR__, "data")

# Load both versions; if a "new" file is missing, skip its block with a warning.
function _safeload(file)
    p = joinpath(PATH_OUT, file)
    isfile(p) ? load_results(p) : nothing
end

println("=" ^ 78)
println("lambda_swap_deltas.jl — post-G=50 vs pre-swap artifacts")
println("=" ^ 78)

# --- Headline bake-off -------------------------------------------------------

bt_new = _safeload("backtest_mc_results.jld2")
bt_old = _safeload("backtest_mc_results.pre_lambda_swap.jld2")
if bt_new === nothing || bt_old === nothing
    @warn "missing backtest_mc_results artifact; skipping headline block"
else
    println("\n--- Headline bake-off (median over 20 seeds) ---")
    println(@sprintf("%-32s | %-30s | %-30s | %-30s",
                     "Strategy", "Sharpe (new vs old)",
                     "MaxDD% (new vs old)", "W_T/W_0 (new vs old)"))
    println("-" ^ 130)
    strategies = collect(keys(bt_new["summary"]))
    for s in strategies
        new_summary = bt_new["summary"][s]
        old_summary = bt_old["summary"][s]
        new_sh  = median(new_summary["sharpe_mc"])
        old_sh  = median(old_summary["sharpe_mc"])
        new_dd  = median(new_summary["max_dd_mc"])
        old_dd  = median(old_summary["max_dd_mc"])
        new_wt  = median(new_summary["W_T_over_W0_mc"])
        old_wt  = median(old_summary["W_T_over_W0_mc"])
        println(@sprintf("%-32s | %+7.3f vs %+7.3f (Δ%+6.3f) | %5.1f%% vs %5.1f%% (Δ%+5.1f) | %5.3f vs %5.3f (Δ%+6.3f)",
                         s, new_sh, old_sh, new_sh - old_sh,
                         100*new_dd, 100*old_dd, 100*(new_dd - old_dd),
                         new_wt, old_wt, new_wt - old_wt))
    end
end

# --- Sweep summary blocks ----------------------------------------------------

function _sweep_block(label, new_file, old_file, grid_key)
    new = _safeload(new_file)
    old = _safeload(old_file)
    if new === nothing || old === nothing
        @warn "missing $(new_file) or $(old_file); skipping $(label) block"
        return
    end
    println("\n--- $label sweep (median over MC seeds, per grid point) ---")
    grid = sort(collect(keys(new["summary"])))
    println(@sprintf("  %-12s | %-25s | %-25s | %-25s",
                     grid_key, "Sharpe (new / old / Δ)",
                     "MaxDD% (new / old / Δ)", "W_T/W_0 (new / old / Δ)"))
    println("  " * "-" ^ 95)
    for g in grid
        sn = new["summary"][g]
        so = haskey(old["summary"], g) ? old["summary"][g] : nothing
        if so === nothing
            println(@sprintf("  %-12s | (no matching grid point in old artifact)", g))
            continue
        end
        sh_n = median(sn["sharpe_mc"])
        sh_o = median(so["sharpe_mc"])
        dd_n = median(sn["max_dd_mc"])
        dd_o = median(so["max_dd_mc"])
        wt_n = median(sn["W_T_over_W0_mc"])
        wt_o = median(so["W_T_over_W0_mc"])
        println(@sprintf("  %-12.4g | %+6.3f / %+6.3f / %+6.3f | %5.1f%% / %5.1f%% / %+5.1f | %5.3f / %5.3f / %+6.3f",
                         g, sh_n, sh_o, sh_n - sh_o,
                         100*dd_n, 100*dd_o, 100*(dd_n - dd_o),
                         wt_n, wt_o, wt_n - wt_o))
    end
    # flag the sweet-spot move
    best_new = argmax(g -> median(new["summary"][g]["sharpe_mc"]), grid)
    grid_old_keys = sort(collect(keys(old["summary"])))
    best_old = argmax(g -> median(old["summary"][g]["sharpe_mc"]), grid_old_keys)
    println(@sprintf("  → max-Sharpe grid point: new = %.4g, old = %.4g (%s)",
                     best_new, best_old, best_new == best_old ? "unchanged" : "MOVED"))
end

_sweep_block("σ_max",        "sigma_max_sweep.jld2",    "sigma_max_sweep.pre_lambda_swap.jld2",    "σ_max")
_sweep_block("w_max",        "w_max_sweep.jld2",        "w_max_sweep.pre_lambda_swap.jld2",        "w_max")
_sweep_block("K_turnover",   "k_turnover_sweep.jld2",   "k_turnover_sweep.pre_lambda_swap.jld2",   "K_turnover")
_sweep_block("cash_revisit", "cash_revisit_sweep.jld2", "cash_revisit_sweep.pre_lambda_swap.jld2", "cash_revisit")
