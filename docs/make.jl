using Pkg
Pkg.develop(path = joinpath(@__DIR__, "..", "code"))
Pkg.instantiate()

using Documenter
using ConstrainedCobbDouglas

makedocs(
    sitename = "Constrained Cobb-Douglas + MPC",
    modules = [ConstrainedCobbDouglas],
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "API" => [
            "SIM"        => "api/sim.md",
            "Allocator"  => "api/allocator.md",
            "MPC"        => "api/mpc.md",
            "Costs"      => "api/costs.md",
            "Tax"        => "api/tax.md",
            "Bandit"     => "api/bandit.md",
            "Backtest"   => "api/backtest.md"],
        "Usage" => [
            "Pipeline" => "usage/pipeline.md",
            "Notebook" => "usage/notebook.md"]],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    warnonly = [:missing_docs])

deploydocs(
    repo = "github.com/varnerlab/modified_cobb_douglas_example.git",
    devbranch = "main")
