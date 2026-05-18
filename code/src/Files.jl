"""
    load_ohlc_jld2(path) -> NamedTuple

Load a lectures-style OHLC JLD2 file. Returns (prices::Matrix, dates::Vector{Date},
tickers::Vector{String}, volumes::Matrix). Schema follows what the lectures
repo writes; missing fields are returned as empty.
"""
function load_ohlc_jld2(path::String)::NamedTuple
    d = load(path)
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
