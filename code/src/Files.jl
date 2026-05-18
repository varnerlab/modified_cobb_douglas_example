"""
    load_ohlc_jld2(path) -> NamedTuple

Load a lectures-style OHLC JLD2 file. Returns `(prices, dates, tickers, volumes)`
where `prices` and `volumes` are `T × K` matrices, `dates` is a `Vector{Date}`
of length `T`, and `tickers` is a `Vector{String}` of length `K`.

Schema actually used by the lectures repo: the file has a single top-level
key `"dataset"` whose value is a `Dict{String, DataFrame}` keyed by ticker.
Each per-ticker `DataFrame` has columns `open, close, high, low, volume,
timestamp, volume_weighted_average_price, number_of_transactions`. The
`timestamp` column is `DateTime` at 05:00:00 UTC; we project to `Date`.

Only tickers with the modal (maximum) number of rows and timestamps matching
the reference timeline are kept — i.e. tickers with partial coverage
(IPO/delisting) are dropped here so that the returned matrices are
rectangular. Downstream code can apply additional universe filters.

Falls back to the older flat-key schema (`"close"`, `"dates"`, etc.) when the
`"dataset"` key is absent, so legacy fixtures still load.
"""
function load_ohlc_jld2(path::String)::NamedTuple
    d = load(path)
    if haskey(d, "dataset") && isa(d["dataset"], Dict)
        ds = d["dataset"]::Dict{String,DataFrame}
        # Pick the modal row count as the reference timeline; this is the set
        # of tickers with full coverage over the file's date range.
        nrows = [nrow(df) for df in values(ds)]
        T_ref = maximum(nrows)
        # Choose a reference ticker with T_ref rows to fix the timeline.
        ref_ticker = first(k for (k, df) in ds if nrow(df) == T_ref)
        ref_ts = ds[ref_ticker].timestamp
        dates  = Date.(ref_ts)

        tickers = String[]
        for (k, df) in ds
            if nrow(df) == T_ref && df.timestamp == ref_ts
                push!(tickers, k)
            end
        end
        sort!(tickers)
        K = length(tickers)
        prices  = Matrix{Float64}(undef, T_ref, K)
        volumes = Matrix{Float64}(undef, T_ref, K)
        for (j, tk) in enumerate(tickers)
            df = ds[tk]
            prices[:, j]  = Float64.(df.close)
            volumes[:, j] = Float64.(df.volume)
        end
        return (prices = prices, dates = dates,
                tickers = tickers, volumes = volumes)
    end

    # Legacy flat-key fallback.
    function pick(keys...)
        for k in keys
            haskey(d, k) && return d[k]
        end
        return nothing
    end
    prices  = pick("close", "prices", "Close")
    dates   = pick("dates", "Date")
    tickers = pick("tickers", "symbols", "Symbol")
    volumes = pick("volume", "Volume")
    return (prices = prices, dates = dates, tickers = tickers, volumes = volumes)
end

"""
    load_sector_map(tickers, csv_path) -> (Dict{String,String}, Vector{String})

Read S&P 500 sector CSV and produce a ticker->sector map plus a list of
unmatched tickers.
"""
function load_sector_map(tickers::Vector{String},
        csv_path::String)::Tuple{Dict{String,String},Vector{String}}
    df = CSV.read(csv_path, DataFrame)
    sym_col = :Symbol
    sec_col = Symbol("GICS Sector")
    lookup = Dict{String,String}(
        row[sym_col] => row[sec_col] for row in eachrow(df))
    sector_of = Dict{String,String}()
    dropped = String[]
    for t in tickers
        if haskey(lookup, t)
            sector_of[t] = lookup[t]
        else
            push!(dropped, t)
        end
    end
    return sector_of, dropped
end

"""
    save_results(path, dict::Dict{String,Any})
"""
function save_results(path::String, d::Dict)
    jldopen(path, "w") do file
        for (k, v) in d
            file[string(k)] = v
        end
    end
    return path
end

"""
    load_results(path) -> Dict
"""
load_results(path::String) = load(path)
