using Test
using ConstrainedCobbDouglas

@testset "ConstrainedCobbDouglas.jl" begin
    include("test_types.jl")
    include("test_sim.jl")
    include("test_allocator.jl")
    include("test_costs.jl")
    include("test_tax.jl")
end
