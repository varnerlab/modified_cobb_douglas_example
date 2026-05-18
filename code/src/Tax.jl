"""
    open_lot!(ledger, ticker, qty, price, date)

Append a new tax lot to the back of the FIFO queue for `ticker`.
"""
function open_lot!(ledger::MyTaxLedger, ticker::String, qty::Int,
        price::Float64, date::Date)
    queue = get!(ledger.lots, ticker, MyTaxLot[])
    push!(queue, MyTaxLot(ticker = ticker, open_date = date,
                          open_price = price, qty = qty))
    return nothing
end

"""
    close_qty!(ledger, ticker, qty_to_close, price, date)

Consume the FIFO queue from the front, accumulating ST/LT P&L based on
the 365-day holding-period boundary. Throws if the requested qty exceeds
available open shares.
"""
function close_qty!(ledger::MyTaxLedger, ticker::String, qty_to_close::Int,
        price::Float64, date::Date)
    queue = get(ledger.lots, ticker, MyTaxLot[])
    remaining = qty_to_close
    while remaining > 0 && !isempty(queue)
        front = queue[1]
        take = min(front.qty, remaining)
        holding_days = (date - front.open_date).value
        pnl = take * (price - front.open_price)
        classification = (holding_days >= 365) ? :lt : :st
        if classification == :lt
            ledger.realized_lt_pnl += pnl
        else
            ledger.realized_st_pnl += pnl
        end
        push!(ledger.closed_lots,
              (ticker = ticker, open_date = front.open_date,
               close_date = date, qty = take, pnl = pnl,
               classification = classification,
               holding_days = holding_days))
        if take == front.qty
            popfirst!(queue)
        else
            front.qty -= take
        end
        remaining -= take
    end
    remaining > 0 && error("close_qty!: attempted to close more shares than open for $ticker")
    return nothing
end

"""
    summarize_after_tax(ledger, rates::NamedTuple) -> NamedTuple

Symmetric tax model: losses generate credits at the category rate.
"""
function summarize_after_tax(ledger::MyTaxLedger,
        rates::NamedTuple)::NamedTuple
    tax_st = rates.st * ledger.realized_st_pnl
    tax_lt = rates.lt * ledger.realized_lt_pnl
    total_tax = tax_st + tax_lt
    realized = ledger.realized_st_pnl + ledger.realized_lt_pnl
    lt_share = realized != 0.0 ? ledger.realized_lt_pnl / realized : 0.0
    hp = [lot.holding_days for lot in ledger.closed_lots]
    return (
        realized_st_pnl = ledger.realized_st_pnl,
        realized_lt_pnl = ledger.realized_lt_pnl,
        tax_st = tax_st,
        tax_lt = tax_lt,
        total_tax = total_tax,
        after_tax_realized_pnl = realized - total_tax,
        lt_share_of_realized = lt_share,
        holding_period_distribution = hp)
end
