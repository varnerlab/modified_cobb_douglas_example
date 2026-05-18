using Test
using Dates
using Random
using Statistics
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

    @testset "should_decide: buy-and-hold strategies only fire at t=1" begin
        state = MyBacktestState()
        state.date_idx = 1
        @test should_decide(EqualWeightStrategy(), state, 1) == true
        @test should_decide(MinVarBuyHoldStrategy(), state, 1) == true
        state.date_idx = 5
        @test should_decide(EqualWeightStrategy(), state, 5) == false
        @test should_decide(MinVarBuyHoldStrategy(), state, 5) == false
    end

    @testset "should_decide: daily strategies fire every day" begin
        state = MyBacktestState()
        state.date_idx = 50
        @test should_decide(UnconstrainedCDStrategy(), state, 50) == true
        @test should_decide(CostAwareMVStrategy(κ = 5.0, c = 0.001), state, 50) == true
    end

    @testset "should_decide: MPC strategies fire on trigger or day 1" begin
        spec = MyMPCSpec(z = 1.96, T = 21, N = 100, D_max = 0.20)
        s5 = CDWithMPCStrategy(spec = spec)
        state = MyBacktestState()
        state.date_idx = 1; state.next_decision_due = false
        @test should_decide(s5, state, 1) == true   # initial allocation
        state.date_idx = 5; state.next_decision_due = false
        @test should_decide(s5, state, 5) == false
        state.next_decision_due = true
        @test should_decide(s5, state, 5) == true
    end

    @testset "run_backtest: EqualWeightStrategy on synthetic prices is reproducible" begin
        using Random
        Random.seed!(1)
        K = 4; n_days = 30
        tickers = ["A","B","C","D"]
        prices = zeros(n_days, K)
        prices[1, :] = [100.0, 50.0, 25.0, 200.0]
        for t in 2:n_days, i in 1:K
            prices[t, i] = prices[t-1, i] * (1.0 + 0.0005 + 0.001 * randn())
        end
        volumes = fill(1.0e9, n_days, K)
        market_prices = vec(mean(prices; dims = 2))
        α = fill(0.0, K); β = fill(1.0, K); σ_ε = fill(0.1, K); σ_m = 0.10
        sim_params = Dict(tickers[i] => (α[i], β[i], σ_ε[i]) for i in 1:K)
        sim_init = Dict(tickers[i] =>
            ewls_init(α[i], β[i], σ_ε[i]; half_life = 21.0, prior_weight = 21.0)
            for i in 1:K)
        env = (tickers = tickers, prices = prices, market_prices = market_prices,
               volumes = volumes, sim_params_init = sim_init,
               σ_m = σ_m, dates = [Date(2025,1,2) + Day(t-1) for t in 1:n_days],
               market_model = nothing, c̄ = 0.05)
        cm = MyCostModel(commission_per_trade = 0.0, half_spread_bps = 5.0,
                         slippage_κ = 0.001,
                         adv = Dict(t => 1e9 for t in tickers))
        rates = (st = 0.37, lt = 0.20)
        res1 = run_backtest(EqualWeightStrategy(), env, cm, rates;
                            B₀ = 100_000.0, rng_seed = 42)
        res2 = run_backtest(EqualWeightStrategy(), env, cm, rates;
                            B₀ = 100_000.0, rng_seed = 42)
        @test res1.wealth_after_cost_pretax == res2.wealth_after_cost_pretax
        @test length(res1.wealth_after_cost_pretax) == n_days
    end
end
