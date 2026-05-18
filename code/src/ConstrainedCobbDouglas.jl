module ConstrainedCobbDouglas

using Clarabel
using CSV
using DataFrames
using Dates
using Distributions
using FileIO
using JLD2
using JuMP
using JumpHMM
using LinearAlgebra
using Random
using SCS
using Statistics
using StatsBase

const _PATH_TO_SRC = dirname(@__FILE__)
const _PATH_TO_DATA = joinpath(_PATH_TO_SRC, "data")

const hmm_simulate = JumpHMM.simulate

# Module includes (populated in later tasks)
include("Types.jl")
# include("SIM.jl")
# include("Allocator.jl")
# include("MPC.jl")
# include("Costs.jl")
# include("Tax.jl")
# include("Bandit.jl")
# include("Backtest.jl")
# include("Files.jl")

end # module
