"""
    materialize_orders(tickers, n_target, n_current, prices, cash_available,
                       cost_model; min_dollar = 100.0) -> Vector{NamedTuple}

Round target shares to integers and emit only orders whose absolute dollar
value exceeds `min_dollar`. The min-order threshold defuses the live engine's
γ-jitter problem (spec §1.3) by suppressing sub-threshold churn.

Returns a vector of (ticker::String, qty::Int) NamedTuples (qty > 0 = buy,
qty < 0 = sell). Cost handling happens at execution time, not here.
"""
function materialize_orders(tickers::Vector{String}, n_target::Vector{Float64},
        n_current::Vector{Float64}, prices::Vector{Float64},
        cash_available::Float64, cost_model::MyCostModel;
        min_dollar::Float64 = 100.0)::Vector{NamedTuple}
    orders = NamedTuple[]
    K = length(tickers)
    for i in 1:K
        q = Int(round(n_target[i] - n_current[i]))
        if q == 0
            continue
        end
        notional = abs(q) * prices[i]
        if notional < min_dollar
            continue
        end
        push!(orders, (ticker = tickers[i], qty = q))
    end
    return orders
end
