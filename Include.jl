# Notebook setup — activate the local package and define data paths.

const _ROOT = pwd()
const _PATH_TO_INPUTS    = joinpath(_ROOT, "code", "src", "data")
const _PATH_TO_ARTIFACTS = joinpath(_ROOT, "scripts", "data")

import Pkg
Pkg.activate(joinpath(_ROOT, "code"))

using ConstrainedCobbDouglas
using JLD2
using DataFrames
using PrettyTables
using Plots
using Statistics
using Dates

function _check_artifact(p)
    if !isfile(p)
        error("Missing artifact: $p\nRun the script that produces it (see scripts/01-05).")
    end
end
