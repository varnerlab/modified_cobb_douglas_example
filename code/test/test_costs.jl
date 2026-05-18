using Test
using ConstrainedCobbDouglas

@testset "Costs module" begin
    cm = MyCostModel(
        commission_per_trade = 0.0,
        half_spread_bps = 5.0,
        slippage_κ = 0.001,
        adv = Dict("AAPL" => 1.0e12))

    @testset "Round-trip half-spread cost" begin
        c_buy  = trade_cost(cm, "AAPL", +100, 100.0)
        c_sell = trade_cost(cm, "AAPL", -100, 100.0)
        # Half-spread on each leg: 5e-4 * 100 * 100 = $5; round-trip = $10
        @test c_buy + c_sell ≈ 10.0 atol=1e-6
    end

    @testset "Slippage scales quadratically with order size" begin
        cm_no_spread = MyCostModel(
            commission_per_trade = 0.0,
            half_spread_bps = 0.0,
            slippage_κ = 0.001,
            adv = Dict("X" => 1.0e6))
        c_100   = trade_cost(cm_no_spread, "X", 100,   50.0)
        c_1000  = trade_cost(cm_no_spread, "X", 1000,  50.0)
        @test c_1000 / c_100 ≈ 100.0 atol=1e-3
    end

    @testset "Zero-share order is zero cost" begin
        @test trade_cost(cm, "AAPL", 0, 100.0) ≈ 0.0
    end

    @testset "Commission is flat" begin
        cm_with_commish = MyCostModel(
            commission_per_trade = 1.0,
            half_spread_bps = 0.0,
            slippage_κ = 0.0,
            adv = Dict("X" => 1.0e9))
        @test trade_cost(cm_with_commish, "X", 1, 10.0) ≈ 1.0
        @test trade_cost(cm_with_commish, "X", 100, 10.0) ≈ 1.0
    end
end
