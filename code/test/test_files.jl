using Test
using JLD2
using CSV
using DataFrames
using ConstrainedCobbDouglas

@testset "Files module" begin
    @testset "load_sector_map returns Dict and detects mismatches" begin
        csv_path = joinpath(@__DIR__, "..", "src", "data", "sp500-sectors.csv")
        sector_of, dropped = load_sector_map(["AAPL", "MSFT", "ZZZZZ_NOT_REAL"], csv_path)
        @test haskey(sector_of, "AAPL")
        @test haskey(sector_of, "MSFT")
        @test "ZZZZZ_NOT_REAL" in dropped
    end

    @testset "save_results / load_results round-trip" begin
        tmp = tempname() * ".jld2"
        d = Dict("a" => [1.0, 2.0], "b" => "hello", "c" => 42)
        save_results(tmp, d)
        d2 = load_results(tmp)
        @test d2["a"] == d["a"]
        @test d2["b"] == d["b"]
        @test d2["c"] == d["c"]
        rm(tmp)
    end
end
