# H2 predictions and gates: rediscover the 3D flows (written 2026-10-27, before any search run)

Stage H2 of `PLAN.md`. Branch `claude/h2-search`. Nothing in this file is changed after the first search run except by a dated amendment at the end.

## What H2 asks

Sample small 3D quadratic flows at random on a coefficient grid, screen them for chaos with the H1 evaluator, and see whether the search re-finds Sprott's 1994 cases A to S (`LITERATURE.md` section C). Also report the hit rate and the size of the space, which sets the budget for H3.

## Definitions fixed now

- **System family.** `x' = P1(x,y,z)`, `y' = P2`, `z' = P3`, each `Pi` a combination of the 10 monomials `1, x, y, z, x^2, xy, xz, y^2, yz, z^2`. 30 coefficients in all. A **term** is one monomial with a nonzero coefficient in one equation (the plan's definition).
- **Coefficient grid.** The 100 nonzero values `-5.0, -4.9, ..., -0.1, 0.1, ..., 5.0`.
- **Sampling distribution D (mine, not Sprott's; his exact rule is not in the pages the H0 session read).** Pick `k` = 5 or 6 with probability 1/2 each; pick `k` of the 30 monomial slots uniformly; give each a coefficient uniformly from the 100 grid values. Cases A to E have 5 terms, F to S have 6.
- **Size of the space** (exact count, `C(30,k) * 100^k`): k = 5: 1.4e15; k = 6: 5.9e17; all k from 1 to 6: 6.0e17; k = 7 would be 2.0e20. The plan says "about 1e20" for Sprott's space; that number matches the 7-term count, not the 6-term count. I could not re-check the paper, so the figure stays unverified, and I use my own counts.
- **Start point.** One fixed start, `(0.05, 0.05, 0.05)`, for the search. A system whose attractor cannot be reached from there is missed. This is a limit of the design, and it is the same in the calibration below.
- **Support class.** The set of monomial slots with nonzero coefficients, taken up to renaming the variables (the 6 permutations of x, y, z applied to the monomials and the equations together). Sign flips of a variable keep the support and keep the grid, so they are inside the class. A search hit **re-finds case X** if it is confirmed chaotic (below) and its support is in the support class of X. Coefficients may differ from the published ones; the plan asks for the cases, and coefficients on a 0.1 grid are matched only by luck.
- **Stages.**
  - Stage 1a (cheap, `vmap`, CPU float64): RK4, `dt = 0.02`, time 40 from the fixed start. Reject if any `|x| > 1e4`, or if the last 20% of the orbit moves less than 1e-3 in every coordinate.
  - Stage 1b: time 600 with a one-vector Benettin estimate of the largest exponent over the last 500. Keep if it exceeds 0.01.
  - Stage 2: H1 spectrum, `T = 5000`, rule v2, sum check (Amendment 4). Keep if labelled "chaotic".
  - Confirmation: H1 spectrum at `T = 20000` from the fixed start and from a second start, both labelled "chaotic". Only confirmed systems count as hits.
- **Hardware.** CPU only (JAX CPU, float64). No GPU run without the owner's word.
- **Sample size N.** Pilot: `N_pilot = 1e5`. Main run: `N = 1e7` unless the measured stage-1 throughput says it would need more than 3 hours of wall time; then N is lowered to what fits, and the value is written in an amendment below before the main run starts. `N = 1e8` only with the owner's approval.

## The calibration that sets what counts as a miss

For every published case X: (a) the probability that a draw from D lands in the support class of X is exact: `0.5 * |class(X)| / C(30,k)`. (b) The chaotic fraction `f_X` inside that class is estimated by drawing 5000 coefficient sets directly on the support (uniform on the grid) and running the full pipeline. The expected number of hits at sample size N is `N * p_X * f_X`. A case with expected hits 3 or more should be found with probability 95% or more.

## Gates

- **G1 (evaluator on all 19 published cases).** Each published coefficient set, run from `(0.05, 0.05, 0.05)` and from two other start points near the origin, `T = 100000`: the 18 dissipative cases are labelled "chaotic" with exponents within `max(0.01, 5%)` of the published ones; case A is flagged conservative. Pass: at least 16 of the 18 match. Every mismatch is reported with a diagnosis (a misread digit from the scanned table is one possible cause).
- **G2 (re-find).** At the main N, every case with expected hits 3 or more is re-found. A case with expected hits below 1 that is not found is a **sampling shortfall**, not an evaluator failure. Expected hits between 1 and 3 are reported as inconclusive. Pass: no case with expected hits 3 or more is missed.
- **G3 (screen recall).** On 300 coefficient sets drawn from each of 6 support classes (F, B, C, S, K, Q), stage 1 keeps at least 90% of the sets that the H1 evaluator (`T = 5000`) labels "chaotic". A screen that drops real chaos would hide hits.
- **G4 (screen rejection).** None of 10 planted non-chaotic systems (fixed point, limit cycle, divergent, conservative, quasi-periodic, plus 5 sign-flipped or permuted copies of published cases with their chaotic coefficients replaced by a clearly periodic set) passes stage 1b. Pass: 0 of 10.
- **G5 (planted bugs).** Three bugs are planted in the new code, one at a time, and a test must fail for each: (1) the support class ignores one of the 6 permutations; (2) stage 1a keeps divergent orbits (the `1e4` test inverted); (3) the monomial-to-equation decode swaps two coefficients. A baseline green suite is required first.
- **G6 (report).** The hit rate (confirmed chaotic per draw, by `k`), the number of distinct support classes found, the size of the space, and the wall time per 1e5 draws are written to `docs/PREDICTIONS-H2.md` (Results). Wall times are not results unless the machine is idle; I will say what else was running.

## Predictions (probability that the statement is true)

- P-H2.1: at least 16 of the 18 dissipative published cases pass G1: 70%. All 18: 45%. The scanned Table I is the weak link.
- P-H2.2: the pilot finds at least one confirmed chaotic system in 1e5 draws: 60%. The chaotic hit rate over D is between 2e-6 and 1e-3: 65%.
- P-H2.3: stage-1 throughput on this CPU is at least 2000 systems per second averaged over a run (most draws die at stage 1a): 50%.
- P-H2.4: G3 passes (recall 90% or more in all six classes): 70%.
- P-H2.5: G4 passes: 90%.
- P-H2.6: at the main N, at least one of the 19 published support classes is re-found: 90%. At least 6 of 19: 45%. All 19: 3%.
- P-H2.7: if a class has expected hits 3 or more, it is found: 90%.
- P-H2.8: the main run finds at least one confirmed chaotic support class with 5 or 6 terms that is **not** in Table I: 55%. If so, I report it as "not in Table I", not as new; H5 does the novelty check.
- P-H2.9: the three planted bugs in G5 are all caught on the first run: 75%.
- P-H2.10: at least one prediction here is missed: 95%.

## What would change my mind

If G1 fails for more than 2 cases, the scanned table is more likely wrong than the evaluator (H1 matched published exponents for B, C, F, S, A and others to 0.001), and I check against a second reading of the table before changing anything. If the pilot hit rate is below 1e-6, N = 1e7 cannot re-find much, and I report the budget needed instead of running a larger search without the owner's word.

## Amendment 1 (2026-10-27, after building the code, before any gate run or search run)

Made after a 20000-draw timing test of the screen (stage 1 only; 3438 draws passed stage 1a, 34 passed stage 1b; 18 of the 18 dissipative published cases passed stage 1) and before G1, G3, G4 or the search were run. Gates G1 and G2 are unchanged. Two feasibility changes:

1. **G3 sample size.** 300 draws per class would give about 3 chaotic sets per class if the chaotic fraction on a support is near 1e-2, which cannot test a 90% recall. G3 now draws 2000 coefficient sets per class (classes F, B, C, S, K, Q) and pools the six classes: pass if the pooled evaluator-chaotic count is at least 30 and stage 1 keeps at least 90% of them. If the pooled count is below 30, more draws are added and the count reported. The ground truth is the H1 evaluator at `T = 5000`, `dt = 0.01`, from the fixed start, run on every draw (including those stage 1 rejects).
2. **G4 stage.** Sprott's case A has a largest exponent of 0.014 (above the 0.01 stage-1 threshold) and is conservative, so it passes stage 1b by design and is rejected at stage 2 by the conservative flag. G4 therefore tests the whole pipeline: none of 10 planted non-chaotic systems is **confirmed** chaotic, and the report names the stage that rejected each. The 10: fixed sink; spiral sink; centre; divergent `x' = 1 + x^2`; Sprott A; Rössler 1976 with c = 2.5, 3, 3.5, 4; Lorenz with rho = 14. Each planted system must first be labelled not-chaotic by the H1 evaluator at `T = 20000` (otherwise it is replaced and the replacement reported).

Machine timing so far: stage 1 about 2300 draws per second in one process (compile included). At that speed `N = 1e7` takes about 70 minutes in one process, so the main run uses several processes; `N = 1e7` stays the plan, and `N = 1e8` (about 12 hours single, about 2 to 3 hours with 6 processes) is only run with the owner's word.

## Amendment 2 (2026-10-27, after the first main run): the sampler used the wrong grid

**What happened.** The code sampled coefficient magnitudes 0.1 to 10.0 with both signs (200 nonzero values), not the registered 0.1 to 5.0 (100 nonzero values). I found it when the hits showed coefficients near 10. The first G3, calibration, pilot and main run (1e7 draws) were all run with this wider grid. No test covered the grid; I had tested the term counts only.

**What I did.** The wider-grid files are kept unchanged in `results/h2_wide_grid/` and are reported as a separate, labelled dataset ("wide grid"; `H2_GRID_MAX=10` reproduces it). The sampler now uses the registered grid (`N_MAG` in `search3d.py`, default 5.0), a test fails if the grid changes (`tests/test_h2_search.py`, which is new), and the planted-bug script `experiments/h2_breakit.py` includes the old grid as bug 4. G3, the calibration, the pilot and the main run are repeated with the registered grid. G1 (published coefficients) and G4 (planted systems) did not use the sampler and are unaffected. No gate or prediction is changed; the predictions are judged on the registered-grid runs, with the wide-grid numbers shown beside them. The size of the registered space is the number in the definitions above (6.0e17 for k up to 6); the wide grid has `C(30,k) * 200^k` = 3.8e19 for k = 6.

**Observation to keep honest about.** The wide grid's space size for k = 6 (3.8e19, 4.0e19 with k = 5 added) is close to the "about 1e20" in the plan, so Sprott's 1e20 may have come from a wider grid or from counting differently than I did. This is a guess; I have not re-read his paper.
