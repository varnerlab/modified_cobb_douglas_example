using Test
using Dates
using ConstrainedCobbDouglas

@testset "Tax module" begin
    @testset "FIFO ordering — closes the older lot first" begin
        led = MyTaxLedger()
        open_lot!(led, "AAPL", 100, 50.0, Date(2025,1,2))
        open_lot!(led, "AAPL", 100, 60.0, Date(2025,2,1))
        close_qty!(led, "AAPL", 50, 70.0, Date(2025,3,1))
        @test length(led.lots["AAPL"]) == 2
        @test led.lots["AAPL"][1].qty == 50   # remainder of the older lot
        @test led.realized_st_pnl ≈ 50 * (70.0 - 50.0)
        @test led.realized_lt_pnl == 0.0
        @test length(led.closed_lots) == 1
    end

    @testset "Partial close shrinks front lot" begin
        led = MyTaxLedger()
        open_lot!(led, "X", 100, 50.0, Date(2025,1,2))
        close_qty!(led, "X", 30, 60.0, Date(2025,2,1))
        @test led.lots["X"][1].qty == 70
        @test led.realized_st_pnl ≈ 30 * 10.0
    end

    @testset "ST/LT boundary at 365 days" begin
        led = MyTaxLedger()
        open_lot!(led, "X", 100, 50.0, Date(2025,1,2))
        close_qty!(led, "X", 50, 60.0, Date(2025,1,2) + Day(364))
        close_qty!(led, "X", 50, 60.0, Date(2025,1,2) + Day(365))
        @test led.realized_st_pnl ≈ 50 * 10.0
        @test led.realized_lt_pnl ≈ 50 * 10.0
    end

    @testset "Over-close throws" begin
        led = MyTaxLedger()
        open_lot!(led, "X", 50, 50.0, Date(2025,1,2))
        @test_throws ErrorException close_qty!(led, "X", 60, 60.0, Date(2025,2,1))
    end
end
