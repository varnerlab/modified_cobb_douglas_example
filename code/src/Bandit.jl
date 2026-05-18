"""
    assign_quotas(sector_groups, K_total) -> Dict{String,Int}

Equal-weight quotas with bonus to largest sectors so Σ q_s = K_total.
"""
function assign_quotas(sector_groups::Dict{String,Vector{Int}},
        K_total::Int)::Dict{String,Int}
    sectors = collect(keys(sector_groups))
    S = length(sectors)
    base = K_total ÷ S
    remainder = K_total - base * S
    sorted = sort(sectors; by = s -> -length(sector_groups[s]))
    q = Dict{String,Int}()
    for (rank, s) in enumerate(sorted)
        q[s] = base + (rank <= remainder ? 1 : 0)
    end
    return q
end

"""
    sample_without_replacement(rng, pool, k) -> Vector{Int}
"""
function sample_without_replacement(rng::AbstractRNG, pool::Vector{Int},
        k::Int)::Vector{Int}
    k >= length(pool) && return copy(pool)
    return shuffle(rng, pool)[1:k]
end

"""
    cd_basket_return(arm_idx, day, horizon, price_matrix, sim_params, tickers, gm_t, lambda; B) -> Float64

Cobb-Douglas-allocated buy-and-hold log return of `arm_idx` (column indices
into `price_matrix`) over `[day, day+horizon]`. Uses analytical CD allocator
with γ = compute_preference_weights at `day`.
"""
function cd_basket_return(arm_idx::Vector{Int}, day::Int, horizon::Int,
        price_matrix::Matrix{Float64}, sim_params::Dict{String,Tuple{Float64,Float64,Float64}},
        tickers::Vector{String}, gm_t::Float64, lambda::Float64;
        B::Float64 = 100_000.0)::Float64
    arm_tickers = tickers[arm_idx]
    γ = compute_preference_weights(sim_params, arm_tickers, gm_t, lambda)
    p_d = price_matrix[day, arm_idx]
    any(p_d .<= 0.0) && return 0.0
    n, cash = solve_unconstrained_cd_analytical(Vector{Float64}(γ), Vector{Float64}(p_d), B)
    p_dh = price_matrix[day + horizon, arm_idx]
    any(p_dh .<= 0.0) && return 0.0
    W_dh = sum(n .* p_dh) + cash
    return log(W_dh / B)
end

"""
    sector_ew_log_return(sector_idx, day, horizon, price_matrix) -> Float64
"""
function sector_ew_log_return(sector_idx::Vector{Int}, day::Int, horizon::Int,
        price_matrix::Matrix{Float64})::Float64
    p_d  = price_matrix[day, sector_idx]
    p_dh = price_matrix[day + horizon, sector_idx]
    (any(p_d .<= 0.0) || any(p_dh .<= 0.0)) && return 0.0
    return log(mean(p_dh ./ p_d))
end

"""
    sector_relative_reward(arm, sector_idx, day, horizon, price_matrix, sim_params,
                           tickers, gm_t, lambda) -> Float64

Cross-sectional alpha: CD-allocated basket log return minus sector EW log return.
"""
function sector_relative_reward(arm::Vector{Int}, sector_idx::Vector{Int},
        day::Int, horizon::Int, price_matrix::Matrix{Float64},
        sim_params::Dict{String,Tuple{Float64,Float64,Float64}},
        tickers::Vector{String}, gm_t::Float64, lambda::Float64)::Float64
    r_basket = cd_basket_return(arm, day, horizon, price_matrix, sim_params,
                                tickers, gm_t, lambda)
    r_sector = sector_ew_log_return(sector_idx, day, horizon, price_matrix)
    return r_basket - r_sector
end

"""
    train_sector_bandit(sector_idx, q, train_offset, train_last, horizon,
                        price_matrix, sim_params, tickers, gm, λ_series;
                        iters, seed, ε_floor) -> NamedTuple

ε-greedy bandit on a single sector. Returns (best_arm, best_mean, rewards,
n_arms, n_unique).
"""
function train_sector_bandit(sector_idx::Vector{Int}, q::Int,
        train_offset::Int, train_last::Int, horizon::Int,
        price_matrix::Matrix{Float64},
        sim_params::Dict{String,Tuple{Float64,Float64,Float64}},
        tickers::Vector{String}, gm::Vector{Float64}, λ_series::Vector{Float64};
        iters::Int, seed::Int, ε_floor::Float64 = 0.05)::NamedTuple
    rng = MersenneTwister(seed)
    N_s = length(sector_idx)
    n_arms = binomial(N_s, q)
    arm_mean = Dict{Vector{Int},Float64}()
    arm_count = Dict{Vector{Int},Int}()
    rewards = zeros(Float64, iters)
    for t in 1:iters
        ε = max(ε_floor,
            t > 1 ? min(1.0, t^(-1/3) * (n_arms * log(t))^(1/3)) : 1.0)
        arm = if rand(rng) < ε || isempty(arm_mean)
            sort(sample_without_replacement(rng, sector_idx, q))
        else
            argmax(arm_mean)::Vector{Int}
        end
        day = rand(rng, train_offset:train_last)
        gm_t = gm[day]; lambda = λ_series[day]
        r = sector_relative_reward(arm, sector_idx, day, horizon,
            price_matrix, sim_params, tickers, gm_t, lambda)
        c = get(arm_count, arm, 0) + 1
        m = get(arm_mean, arm, 0.0)
        arm_mean[arm] = m + (r - m) / c
        arm_count[arm] = c
        rewards[t] = r
    end
    best_arm = argmax(arm_mean)
    return (best_arm = best_arm,
            best_mean = arm_mean[best_arm],
            rewards = rewards,
            n_arms = n_arms,
            n_unique = length(arm_mean))
end
