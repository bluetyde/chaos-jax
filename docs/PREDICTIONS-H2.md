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

## Results (registered grid, written 2026-10-27 after the runs; no gate or prediction above was changed)

Run: `calib 6 5000`, `search pilot 5 100000`, `search main 6 1e7` (chunk 20000, seeds 1000-1499), `g3 6`; CPU only, 6 worker processes, branch `claude/h2-search` at the commit that holds `search3d.py` with `N_MAG` = 50. Files: `results/h2_calib.*`, `h2_pilot.*`, `h2_main.jsonl/log/_summary.json`, `h2_g3.*`. The wide-grid run is separate (`results/h2_wide_grid/`) and is not used for any gate.

**Totals (main, 1e7 draws).** Stage 1a kept 2,327,312; stage 1b kept 27,355; stage 2 (T = 5000) labelled 610 chaotic; **472 confirmed** (T = 20000 from two starts). Hit rate 4.72e-5 per draw. By term count: k = 5 gave 124 hits, k = 6 gave 348 (2.5e-5 and 7.0e-5 per draw, 5e6 draws each). Distinct support classes with a confirmed hit: **266**; 3 of them are published classes, **263 are not in Table I**. Space size (registered grid, up to 6 terms) about 6.0e17; the 1e7 draws cover about 1.7e-11 of it.

**Expected hits per published class** (N x p_support x f_chaotic from `h2_calib.json`, N = 1e7):

| class | expected | found (hits) |
|---|---|---|
| B | 10.9 | 8 |
| C | 10.0 | 15 |
| D | 2.9 | 2 |
| E | 1.5 | 0 |
| A | 0 (rejected at stage 1b by design) | 0 |
| F to S (14 classes, k = 6) | 0.01 to 0.19 each, 1.3 in total | 0 |

**Gates.**
- G1: passed earlier (18 of 18 dissipative cases match the published exponents, A flagged conservative).
- G2: **pass.** The only classes with expected hits of 3 or more are B and C, and both were found. D (2.9) was found twice, E (1.5) is inconclusive by the rule above, and the 14 k = 6 classes with expected hits below 1 not found are sampling shortfalls, not evaluator failures.
- G3: **pass.** Pooled over F, B, C, S, K, Q at 2000 draws each: 245 evaluator-chaotic sets, stage 1 kept 245 (recall 100%; the pooled count is above the 30 required).
- G4: passed earlier (0 of 10 planted systems confirmed chaotic). G5: all 4 planted bugs caught on the first run, tree clean after.
- G6: written here. Wall time: 8751 s for 1e7 draws with 6 workers, about 87 s per 1e5 draws. Not a clean timing: other agents were running on this machine during the run, and I did not check for idle CPU, so treat it as an upper bound.

**Predictions judged.**
- P-H2.1 true (18 of 18). P-H2.2 true (pilot: 1 confirmed in 1e5 draws; main rate 4.7e-5 is inside 2e-6 to 1e-3).
- P-H2.3 **missed on its wording.** Stage 1 alone ran at about 2300 draws per second in one process (Amendment 1 timing), but averaged over the whole pipeline the run did about 1140 draws per second across 6 workers, because stage 2 dominates. The prediction said "averaged over a run", so I count it as a miss.
- P-H2.4 true. P-H2.5 true. P-H2.9 true.
- P-H2.6: at least 1 of 19 re-found: true (3). At least 6 of 19: **missed** (3 found). All 19: false, as predicted (3%).
- P-H2.7 true (B and C found; D, expected 2.9, was also found).
- P-H2.8 true: 263 classes with 5 or 6 terms are not in Table I. This is "not in Table I", not "new"; H5 does the novelty check.
- P-H2.10 true (two misses above).

**Wide grid, for comparison.** The same 1e7 draws on the unregistered grid (magnitudes up to 10) gave a different hit rate and class mix; numbers are in `results/h2_wide_grid/h2_main_summary.json`. The two sets are different samples of different spaces and are not pooled.

**Caveats before anyone calls any hit a finding.**
1. The 263 other classes are support classes that the evaluator labels chaotic at T = 20000 from two starts. None has been checked for being a linear-change-of-variables or time-rescaling copy of a known flow (a shift of origin changes the support, so two supports can describe one flow). H5 does that.
2. The strongest hits have large coefficients (up to 5) and very negative third exponents; these are dissipative and bounded in the runs, but only two starts were tried.
3. The three published classes found are the ones with the highest on-support chaos fraction (B, C, D); the k = 6 published classes are at expected hits near 0.1 each, so 1e7 draws cannot test whether the search finds them. A run 100 times larger (1e9, about 10 days at this speed on 6 CPU workers, or a GPU stage 2) would be needed, and I have not started either.
