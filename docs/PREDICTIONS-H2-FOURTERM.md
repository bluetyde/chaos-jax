# H2 four-term search: predictions and gates (written 2026-10-28, before any code or run)

## Question
Is there a 3D quadratic flow with only 4 nonzero terms (out of 30 slots) that is numerically chaotic? The smallest published 3D chaotic flows have 5 terms (Sprott 1994 cases A-E; `LITERATURE.md` section C). The H2 random search drew only k = 5 and 6, so 4 terms was never searched.

## Why not sample
There are 27,405 supports of 4 slots, times 100 nonzero grid values per slot (0.1..5.0, both signs): 2.7e12 vectors. Sampling cannot cover that. But coefficients are not independent: rescaling time (t to tau t) and each variable (x_i to s_i x_i) with positive factors maps the flow to another flow with coefficients c' = tau c s^e / s_i (e = the monomial's exponents), and multiplies every exponent by tau. Chaos is unchanged. Per support, the log-magnitudes of the 4 coefficients can be moved by the 4 free scalings. If the 4x4 matrix M (row per term: 1, e - unit vector of its equation) has rank 4 all magnitudes can be set to 1, so only the 16 sign patterns remain. If it has rank 3 or 2, one or two scale-invariant combinations remain and are scanned on a grid. Sign flips and variable renamings (the 48-element group) are not removed in advance; they just repeat work.

## Method
- Supports with every one of the three equations nonempty (a variable with no equation is constant and the flow is 2D): 13,476. Rank 4: 8,346; rank 3: 4,881; rank 2: 273.
- Rank-deficient supports: each invariant t on a log grid of 13 values, ln t from -3 to +3 in steps of 0.5 (factor 20 each side, wider than the registered grid reaches). Rank 4: 1 magnitude vector; all supports: 16 sign patterns.
- Total about 1.9 million systems. Pipeline as H2 without the float32 pass: stage 1a/1b screen, boundedness filter, float64 stage 2 (T = 5000), confirmation (T = 20000, two starts), then H1 Amendment 5 (expanding orbits rejected, inside `classify`).
- Code: `src/hyperchaos/enumerate.py`, runner `experiments/h2_fourterm.py`.

## Controls (run before the search)
- **C1 (algebra).** For a known chaotic system and random positive scalings, the scaled system's exponents are tau times the original's, and its support is unchanged. Unit test.
- **C2 (rediscovery).** The same enumerator, run over the support of each of the 5-term Sprott cases A-E (k = 5; one scale invariant per support when the rank is 4), must rediscover each case: at least one confirmed chaotic hit on that support.
- **C3 (planted bug).** Break the enumerator (use the wrong scaling formula) and show a test fails.

## Gates
- **GF1.** C1 passes; C2 finds at least 4 of the 5 cases (A-E); C3 caught.
- **GF2 (report).** Number of systems screened, stage-1a survivors, stage-2 survivors, confirmed hits, with counts by rank. If a 4-term hit appears: re-run it at dt 0.005 and 0.0025 and from the two starts, and check the sum of exponents is negative, before calling it anything.

## Predictions (probability the statement is true)
- P-F1: no 4-term confirmed hit: 75%. At least one: 25%. At least 10: 4%.
- P-F2: C2 finds all 5 cases A-E: 55%; at least 4: 80%.
- P-F3: run time under 30 minutes on 6 workers: 70%.
- P-F4: stage-1a survivors are below 5% of the systems screened: 70% (most 4-term flows are too simple to move or blow up).
- P-F5: if any 4-term hit exists, it is at rank 3 or 2 (a scale invariant is needed), not rank 4: 70%.
- P-F6: at least one prediction is missed: 60%.

## Not claimed
A "no 4-term chaotic flow exists" statement. A negative result covers: quadratic 3D flows with these monomials, rank-4 supports up to scaling exactly, rank-3 and rank-2 supports only on the 13-point invariant grid, start points START and START2 only, classifier rule v2 with Amendment 5, dt 0.01 for the confirmation. Bounded solutions that need other starting points are not searched.

## Erratum (2026-10-28, before any run)
The support total above says 13,476; the three rank counts it lists (8,346 + 4,881 + 273) sum to 13,500, which is what the code enumerates. The rank counts are unchanged; only the total was mis-added. No gate or prediction depends on it.

## Results (2026-10-28, written after the runs; no gate or prediction above was changed)

Files: `src/hyperchaos/enumerate.py`, `tests/test_enumerate.py`, `experiments/h2_fourterm.py`, `h2_fourterm_verify.py`, `h2_fourterm_checks.py`, `results/h2_fourterm_{control*,search,verify,checks}.*`, `results/h1_breakit_enum.log`. float64, 6 workers, Windows, CPU. Search wall time 2,608 s (43 min).

**Headline: no dissipative 4-term chaotic flow was found.** 1,886,976 systems screened (rank 4: 133,536; rank 3: 1,015,248; rank 2: 738,192). Stage 1a survivors 688,760 (36.5%), stage 1b 17,382, boundedness 1,382, stage 2 (dissipative, chaotic, T = 5000): **0**. Five conservative candidates were recorded separately.

**Controls.** C1 passes (scaling multiplies the exponents by tau, support unchanged; 4 tests). C3 passes: planted bug 6 caught (bugs 1-6 all caught, 53 baseline tests). C2: on the fine invariant grid (step 0.1) the enumerator rediscovers B, C, D, E as dissipative chaotic (4, 4, 18, 36 confirmed systems) and A (Nosé-Hoover) only as a conservative candidate (51 systems). A is missed by the dissipative rule by design (the project's own control `N-cons`); my first control run did not record conservative candidates, so it showed 4 of 5, and I changed the worker to record them and reran case A. GF1 passes (at least 4 of 5 either way).
**Coarse-grid control (added after the search, not pre-registered).** At the step 0.5 used for rank 3 and rank 2 supports, the enumerator finds B, C, D but **not E** (E's invariant is -0.347; its chaotic window is narrower than the grid step). So for the 5,154 rank-3 and rank-2 supports the negative result is weaker than for the rank-4 supports.

**The five conservative candidates are not chaotic.** All five have a quadratic first integral: d/dt of y^2 - z^2 (candidates 0 and 1), x^2 - z^2 (2 and 3) or x^2 - y^2 (4) is zero to 1e-13 at 200 random points each (`results/h2_fourterm_checks.log`). A flow with a first integral moves on a 2D surface, and a bounded 2D autonomous flow cannot be chaotic. The numerical evidence agrees: the second start is quasi-periodic for all five (first exponent 0.0004 to 0.0005); the first start gives exponents that change by a factor of 3 to 10 between dt 0.01, 0.005 and 0.0025 (candidates 2 and 3: 9.2, 1.07, 9.8, or inconsistent with |x| up to 2e4). The small positive exponents (0.01 to 0.03) of candidates 0, 1 and 4 are finite-time values near a separatrix of the reduced flow, not a chaotic attractor. This shows the classifier can label a conservative flow with a first integral as chaotic; the dissipative rule (Amendment 5 and the conservative flag) keeps these out of the hit lists, but a conservative-search would need a first-integral check.

**Predictions.** P-F1 hit (no 4-term confirmed hit; the 5 conservative candidates are explained). P-F2 **"all 5" missed** on the pre-registered definition (4 of 5 on the first run; "at least 4" hit). P-F3 **missed** (43 min against under 30). P-F4 **missed**: 36.5% of systems pass stage 1a, not under 5% (many 4-term flows are bounded and moving; they are not chaotic, but they are not trivially simple). P-F5 not applicable (no hits). P-F6 hit.

**What this does and does not say.** In this search space (quadratic 3D flows with 4 terms, every equation nonempty, positive scalings removed, start points START and START2, classifier rule v2 with Amendment 5) no dissipative chaotic flow appeared. Rank-4 supports (8,346 of 13,500) are covered exactly up to scaling and sign. Rank-3 and rank-2 supports are covered only at invariant step 0.5, and case E shows a narrow window can slip through. This is consistent with Sprott's finding that the smallest chaotic 3D flows have 5 terms; it does not prove there is no 4-term chaotic flow.
