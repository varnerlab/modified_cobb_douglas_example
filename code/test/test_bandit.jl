using Test
using Random
using Statistics
using ConstrainedCobbDouglas

@testset "Bandit module" begin
    @testset "assign_quotas sums to K_basket for uniform 11 sectors" begin
        sector_groups = Dict("S$i" => collect((i-1)*40+1:i*40) for i in 1:11)
        q = assign_quotas(sector_groups, 22)
        @test sum(values(q)) == 22
        @test all(values(q) .== 2)
    end

    @testset "sample_without_replacement picks distinct elements" begin
        rng = MersenneTwister(1)
        pool = collect(1:50)
        s = ConstrainedCobbDouglas.sample_without_replacement(rng, pool, 5)
        @test length(s) == 5
        @test length(unique(s)) == 5
        @test all(x in pool for x in s)
    end

    @testset "ε decay is monotone non-increasing" begin
        n_arms = 1000
        ε_floor = 0.05
        ε(t) = max(ε_floor, t > 1 ? min(1.0, t^(-1/3) * (n_arms * log(t))^(1/3)) : 1.0)
        seq = [ε(t) for t in 2:5000]
        @test all(seq[2:end] .<= seq[1:end-1] .+ 1e-12)  # non-increasing
        @test all(seq .>= ε_floor)                       # never below floor
        @test seq[1] <= 1.0                              # capped at 1
    end
end
