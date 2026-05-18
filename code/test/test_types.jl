using Test
using Dates
using ConstrainedCobbDouglas

@testset "Type construction" begin
    @testset "MySIMParameterEstimate" begin
        est = MySIMParameterEstimate()
        est.ticker = "AAPL"; est.α = 0.05; est.β = 1.2; est.σ_ε = 0.20; est.r² = 0.65
        @test est.ticker == "AAPL"
        @test est.α == 0.05
    end

    @testset "MyEWLSState" begin
        s = MyEWLSState()
        s.α = 0.0; s.β = 1.0; s.σ_ε = 0.2
        s.Sw = 1.0; s.Swx = 0.0; s.Swy = 0.0
        s.Swxx = 1.0; s.Swxy = 1.0; s.Swyy = 1.0; s.η = 0.99
        @test s.β == 1.0
    end

    @testset "MyConstrainedCDProblem and Result" begin
        p = MyConstrainedCDProblem(
            γ = [0.5, 0.3], p = [100.0, 50.0], B = 10000.0,
            Σ = [0.04 0.01; 0.01 0.09], σ_max = 0.15,
            K_turnover = 1000.0, w_max = 0.20,
            n_prev = [10.0, 20.0], c̄ = 0.05)
        @test p.B == 10000.0
        @test length(p.γ) == 2

        r = MyConstrainedCDResult(
            n = [50.0, 100.0], w = [0.5, 0.5],
            unallocated_budget = 0.0,
            status = :optimal,
            objective = 1.234)
        @test r.status == :optimal
    end

    @testset "MyMPCSpec, MyMPCProjection, MyMPCTrigger" begin
        spec = MyMPCSpec(z = 1.96, T = 21, N = 1000, D_max = 0.08)
        @test spec.T == 21

        proj = MyMPCProjection(
            μ = ones(21), σ = 0.01 * ones(21),
            V₀ = 100_000.0, paths = ones(1000, 21),
            decision_date_idx = 1,
            closed_form_μ = ones(21), closed_form_σ = 0.01 * ones(21),
            divergence_warning = false)
        @test length(proj.μ) == 21

        trig = MyMPCTrigger(fired = false, reason = :in_spec, τ = 5)
        @test trig.reason == :in_spec
    end

    @testset "MyCostModel" begin
        cm = MyCostModel(
            commission_per_trade = 0.0,
            half_spread_bps = 5.0,
            slippage_κ = 0.001,
            adv = Dict("AAPL" => 1.0e7))
        @test cm.half_spread_bps == 5.0
    end

    @testset "MyTaxLot and MyTaxLedger" begin
        lot = MyTaxLot(ticker = "AAPL", open_date = Date(2025,1,2),
                       open_price = 150.0, qty = 100)
        @test lot.qty == 100

        ledger = MyTaxLedger()
        @test isempty(ledger.lots)
        @test ledger.realized_st_pnl == 0.0
    end

    @testset "Strategy types" begin
        s1 = EqualWeightStrategy()
        s2 = MinVarBuyHoldStrategy()
        s3 = UnconstrainedCDStrategy()
        s4 = CostAwareMVStrategy(κ = 5.0, c = 0.0005)
        spec = MyMPCSpec(z = 1.96, T = 21, N = 1000, D_max = 0.08)
        s5 = CDWithMPCStrategy(spec = spec)
        s6 = ConstrainedCDWithMPCStrategy(spec = spec, σ_max = 0.12,
                                         K_turnover = 10_000.0, w_max = 0.20)
        @test isa(s1, MyAllocationStrategy)
        @test isa(s6, MyAllocationStrategy)
        @test s4.κ == 5.0
        @test s6.σ_max == 0.12
    end
end
