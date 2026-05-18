using Test
using ConstrainedCobbDouglas

@testset "ConstrainedCobbDouglas.jl" begin
    include("test_types.jl")
    include("test_sim.jl")
end
