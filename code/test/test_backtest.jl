using Test
using Dates
using ConstrainedCobbDouglas

@testset "Backtest module" begin
    @testset "materialize_orders rounds and respects min-order" begin
        cm = MyCostModel(commission_per_trade = 0.0, half_spread_bps = 5.0,
                         slippage_κ = 0.001, adv = Dict("A"=>1e9, "B"=>1e9))
        n_target = [99.6, 50.4]
        n_current = [100.0, 50.0]
        prices = [100.0, 50.0]
        orders = materialize_orders(["A","B"], n_target, n_current, prices,
                                    1e6, cm; min_dollar = 1000.0)
        # qty deltas: -0.4, +0.4 → after round: 0, 0 → all suppressed by min_dollar
        @test isempty(orders)
    end

    @testset "materialize_orders generates an order above threshold" begin
        cm = MyCostModel(commission_per_trade = 0.0, half_spread_bps = 5.0,
                         slippage_κ = 0.001, adv = Dict("A"=>1e9))
        orders = materialize_orders(["A"], [120.0], [100.0], [100.0],
                                    1e6, cm; min_dollar = 100.0)
        @test length(orders) == 1
        @test orders[1].ticker == "A"
        @test orders[1].qty == 20
    end
end
