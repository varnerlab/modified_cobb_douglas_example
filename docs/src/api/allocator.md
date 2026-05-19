# Allocator API

## Problem and result types

```@docs
MyConstrainedCDProblem
MyConstrainedCDResult
```

## Strategy types

```@docs
MyAllocationStrategy
EqualWeightStrategy
MinVarBuyHoldStrategy
UnconstrainedCDStrategy
CostAwareMVStrategy
CDWithMPCStrategy
ConstrainedCDWithMPCStrategy
```

## Solvers

```@docs
solve_constrained_cd
solve_unconstrained_cd_analytical
equal_weight_target
solve_minvar_buyhold
solve_cost_aware_mv
```
