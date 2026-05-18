# Theory

Restates the math from the implementation design doc, sections 4 (constrained
CD), 5 (MPC), and 7 (bandit). See `docs/superpowers/specs/2026-05-17-constrained-cobb-douglas-design.md`
for the full theory recap; this page links to the API references for each
function that implements a piece of the math.

## Constrained Cobb-Douglas

Objective and constraints — see [`solve_constrained_cd`](api/allocator.md).

## MPC forward projection

Step-by-step under continuous compounding — see [`forward_project`](api/mpc.md).

## Per-sector bandit

Algorithm — see [`train_sector_bandit`](api/bandit.md).
