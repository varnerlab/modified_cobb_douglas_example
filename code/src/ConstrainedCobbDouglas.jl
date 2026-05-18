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

export MySIMParameterEstimate, MyEWLSState
export MyConstrainedCDProblem, MyConstrainedCDResult
export MyMPCSpec, MyMPCProjection, MyMPCTrigger
export MyCostModel
export MyTaxLot, MyTaxLedger
export MyAllocationStrategy, EqualWeightStrategy, MinVarBuyHoldStrategy,
       UnconstrainedCDStrategy, CostAwareMVStrategy,
       CDWithMPCStrategy, ConstrainedCDWithMPCStrategy
export MyBanditConfig, MyBanditResult
export MyBacktestState, MyBacktestResult
export estimate_sim, build_sim_covariance
export compute_market_growth, compute_ema, compute_lambda, compute_preference_weights

include("SIM.jl")
# include("Allocator.jl")
# include("MPC.jl")
# include("Costs.jl")
# include("Tax.jl")
# include("Bandit.jl")
# include("Backtest.jl")
# include("Files.jl")

end # module
