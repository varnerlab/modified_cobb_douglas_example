# --- Analytical unconstrained Cobb-Douglas (vendored from Compute.jl
# allocate_cobb_douglas, lines 505-541). Used as strategy 3 and strategy 5
# allocator and as the loose-constraint reference for the JuMP solver. ---

"""
    solve_unconstrained_cd_analytical(γ, p, B; ε = 1e-3) -> (shares, cash)

Closed-form Cobb-Douglas allocator. Non-preferred assets (γᵢ ≤ 0) get
ε shares; preferred assets get proportional allocation.
"""
function solve_unconstrained_cd_analytical(γ::Vector{Float64}, p::Vector{Float64},
        B::Float64; ε::Float64 = 1e-3)::Tuple{Vector{Float64},Float64}
    K = length(γ)
    preferred = findall(γ .> 0)
    non_preferred = findall(γ .<= 0)
    shares = zeros(K)
    remaining_B = B
    for i in non_preferred
        shares[i] = ε
        remaining_B -= ε * p[i]
    end
    cash = 0.0
    if !isempty(preferred) && remaining_B > 0
        γ_bar = sum(γ[preferred])
        for i in preferred
            shares[i] = (γ[i] / γ_bar) * (remaining_B / p[i])
        end
    else
        cash = remaining_B
    end
    return (shares, cash)
end

"""
    equal_weight_target(K::Int) -> Vector{Float64}

Equal-weight target weights summing to 1.0.
"""
equal_weight_target(K::Int)::Vector{Float64} = fill(1.0 / K, K)

"""
    solve_minvar_buyhold(Σ, bounds) -> Vector{Float64}

Solve min wᵀΣw  s.t.  Σwᵢ = 1, bounds[i,1] ≤ wᵢ ≤ bounds[i,2].
"""
function solve_minvar_buyhold(Σ::Matrix{Float64},
        bounds::Matrix{Float64})::Vector{Float64}
    K = size(Σ, 1)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, w[1:K])
    @constraint(model, [i in 1:K], w[i] >= bounds[i, 1])
    @constraint(model, [i in 1:K], w[i] <= bounds[i, 2])
    @constraint(model, sum(w) == 1.0)
    @objective(model, Min, w' * Σ * w)
    optimize!(model)
    return value.(w)
end

# --- Constrained Cobb-Douglas solver (spec §4) ---

"""
    solve_constrained_cd(problem::MyConstrainedCDProblem) -> MyConstrainedCDResult

Solve max Σ γᵢ log(nᵢ) s.t. budget, covariance, turnover, and concentration
constraints (spec §4). Uses Clarabel via JuMP; falls back to SCS if Clarabel
returns a non-`OPTIMAL` status.

Non-preferred assets (γᵢ ≤ 0) are pinned at ε = 1e-3 shares; the optimizer
runs over the preferred subset only (keeps the objective concave).
"""
function solve_constrained_cd(problem::MyConstrainedCDProblem;
        ε::Float64 = 1e-3)::MyConstrainedCDResult
    γ = problem.γ; p = problem.p; B = problem.B
    Σ = problem.Σ; σ_max = problem.σ_max
    K_turnover = problem.K_turnover; w_max = problem.w_max
    n_prev = problem.n_prev; c̄ = problem.c̄
    K = length(γ)
    preferred = findall(γ .> 0)
    non_pref = findall(γ .<= 0)

    if isempty(preferred)
        n_full = zeros(K)
        for i in non_pref; n_full[i] = ε; end
        return MyConstrainedCDResult(
            n = n_full,
            w = (n_full .* p) ./ B,
            unallocated_budget = B - sum(n_full .* p),
            duals = (σ_max = 0.0, turnover = 0.0, w_max = 0.0),
            status = :no_preferred,
            objective = 0.0)
    end

    # Regularize Σ for Cholesky
    Σ_reg = Σ + 1e-8 * I(K)
    L = cholesky(Σ_reg).L

    pinned_cost = sum(ε * p[i] for i in non_pref; init = 0.0)
    B_eff = B - pinned_cost

    Kp = length(preferred)
    p_p = p[preferred]
    n_prev_p = n_prev[preferred]

    function build_model(opt)
        m = Model(opt)
        set_silent(m)
        @variable(m, n[1:Kp] >= 1e-8)
        @variable(m, t[1:Kp])
        # log hypograph via exponential cone: t ≤ log(n)  ⇔  (t, 1, n) ∈ ExpCone
        for k in 1:Kp
            @constraint(m, [t[k], 1.0, n[k]] in MOI.ExponentialCone())
        end
        @objective(m, Max, sum(γ[preferred[k]] * t[k] for k in 1:Kp))

        # Budget
        @constraint(m, sum(n[k] * p_p[k] for k in 1:Kp) <= B_eff)

        # Full weight vector (preferred + pinned non-preferred)
        @expression(m, w_full[i = 1:K],
            (i in preferred) ?
                n[findfirst(==(i), preferred)] * p[i] / B :
                ε * p[i] / B)

        # Concentration cap
        for i in 1:K
            @constraint(m, w_full[i] <= w_max)
        end

        # Covariance budget via SOC: ||Lᵀ w|| ≤ σ_max
        Lt = Matrix(L')
        @constraint(m, [σ_max; Lt * collect(w_full)] in SecondOrderCone())

        # Turnover budget (l1) — slack vars on preferred only; non-preferred churn ignored
        # since their position changes ε → ε (zero) under the same regime.
        # Skip slacks entirely when c̄ = 0 (turnover unpriced; constraint is vacuous and
        # the redundant 0 ≤ K rows cause Clarabel scaling pathologies).
        if c̄ > 0
            @variable(m, u[1:Kp] >= 0)
            for k in 1:Kp
                @constraint(m, u[k] >= n[k] - n_prev_p[k])
                @constraint(m, u[k] >= n_prev_p[k] - n[k])
            end
            @constraint(m, c̄ * sum(u) <= K_turnover)
        end
        return m, n, t
    end

    # Try Clarabel first
    clarabel_opt = optimizer_with_attributes(Clarabel.Optimizer,
        "tol_gap_abs" => 1e-9, "tol_gap_rel" => 1e-9, "tol_feas" => 1e-9,
        "max_iter" => 500)
    m, nvar, tvar = build_model(clarabel_opt)
    optimize!(m)
    status = termination_status(m)
    accepted = (status == MOI.OPTIMAL) || (status == MOI.ALMOST_OPTIMAL)

    if !accepted
        # Fall back to SCS
        m, nvar, tvar = build_model(SCS.Optimizer)
        optimize!(m)
        status = termination_status(m)
        accepted = (status == MOI.OPTIMAL) || (status == MOI.ALMOST_OPTIMAL)
    end

    if !accepted
        return MyConstrainedCDResult(
            n = copy(n_prev),
            w = (n_prev .* p) ./ B,
            unallocated_budget = max(0.0, B - sum(n_prev .* p)),
            duals = (σ_max = 0.0, turnover = 0.0, w_max = 0.0),
            status = :solver_failed,
            objective = 0.0)
    end

    n_p = value.(nvar)
    n_full = zeros(K)
    for (k, i) in enumerate(preferred); n_full[i] = n_p[k]; end
    for i in non_pref; n_full[i] = ε; end
    obj = objective_value(m)
    w_full = (n_full .* p) ./ B
    return MyConstrainedCDResult(
        n = n_full,
        w = w_full,
        unallocated_budget = max(0.0, B - sum(n_full .* p)),
        duals = (σ_max = 0.0, turnover = 0.0, w_max = 0.0),  # filled in Task 10
        status = :optimal,
        objective = obj)
end

"""
    solve_cost_aware_mv(γ, Σ, w_prev; κ, c) -> Vector{Float64}

Strategy 4: max γᵀw - (κ/2) wᵀΣw - c·‖w - w_prev‖₁  s.t. Σwᵢ = 1, wᵢ ≥ 0.
"""
function solve_cost_aware_mv(γ::Vector{Float64}, Σ::Matrix{Float64},
        w_prev::Vector{Float64}; κ::Float64, c::Float64)::Vector{Float64}
    K = length(γ)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, w[1:K] >= 0)
    @variable(model, u[1:K] >= 0)
    @constraint(model, sum(w) == 1.0)
    for i in 1:K
        @constraint(model, u[i] >=  w[i] - w_prev[i])
        @constraint(model, u[i] >= -(w[i] - w_prev[i]))
    end
    @objective(model, Max, sum(γ[i] * w[i] for i in 1:K)
                          - (κ / 2.0) * w' * Σ * w
                          - c * sum(u))
    optimize!(model)
    return value.(w)
end
