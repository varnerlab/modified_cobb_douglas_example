"""
    trade_cost(model::MyCostModel, ticker, q_signed, price) -> Float64

Per-fill cost: half-spread + linear-impact slippage + flat commission.
`q_signed` positive for buy, negative for sell.
"""
function trade_cost(model::MyCostModel, ticker::String, q_signed::Int,
        price::Float64)::Float64
    if q_signed == 0
        return 0.0
    end
    q = abs(q_signed)
    adv_t = get(model.adv, ticker, 1.0e9)   # default huge ADV → ~no slippage
    half_spread_cost = (model.half_spread_bps * 1e-4) * price * q
    slippage_cost    = model.slippage_κ * (q / adv_t) * price * q
    return half_spread_cost + slippage_cost + model.commission_per_trade
end
