# H2/H4: structure keys and an exhaustive 5-term pass as a speedup candidate: predictions and gates (written 2026-10-28, before any code or run)

Owner's instruction: build the keys and the exhaustive 5-term pass; enable it only if it speeds the work up, otherwise leave it unenabled like the other speedup attempts (the default stays random search).

## What is built
1. **Keys** (`src/hyperchaos/keys.py`). Structure key = the term-set and signs, minimised over the 48 renamings and sign flips. Parameter key = the scale-invariant coordinates (`enumerate.invariants`) of the representative that attains the minimum. Used to count families among hits and to compare with Sprott's cases A-S.
2. **Exhaustive 5-term pass** (`experiments/h2_enum5.py`). All 16,295 renaming classes of 5-term supports (every equation nonempty); signs reduced by the 8 sign flips (one representative per flip orbit); the scale invariants on a grid of 13 points (ln t from -3 to 3, step 0.5) per invariant; the same screen and classifier as the night run (stage 1a, stage 1b, boundedness, stage 2 T = 5000, confirmation T = 20000 with two starts, Amendment 5), start points START and START2 as before. Resumable chunk files, stall timeout of 30 minutes.

## Baseline for the comparison
The night run (`results/h2_night1_*`) found 4,634 hits on 1,093 term-sets: 38 five-term sets and 1,055 six-term sets (counted by canonical support). Its cost for 5-term sets is not recorded separately, so I measure it: a random 5-term-only sample (k = 5, 10 seeds of 20,000 draws, same pipeline, same machine, run in the same session) gives seconds per draw. The night run drew about 5e7 five-term systems (half of 1e8; the sampler picks k = 5 or 6 with equal probability). Baseline cost = per-draw cost x 5e7. This is an extrapolation and is labelled as one.

## Gates for enabling
- **GE1 (recall).** The exhaustive pass finds at least 37 of the 38 five-term families the night run found (families identified by canonical support, 5-term only).
- **GE2 (speed).** Wall time of the exhaustive pass is at most half the extrapolated baseline cost of the random run (a speedup of at least 2x on equal footing).
- **GE3 (controls).** Sprott cases A-E are all found by the pass (B-E as dissipative chaotic, A as conservative chaotic, as in the 4-term control).
- **GE4 (keys).** A unit test shows the structure and parameter keys are unchanged by a random positive rescaling plus renaming plus sign flip of a known system, and differ for Sprott cases A-E; a planted bug (key built without the sign flips) fails a test.
Enabled only if GE1-GE4 all hold. If not: the code stays, the default stays random search, and the result is reported as a miss.

## Predictions (probability the statement is true)
- P-E1: GE1 holds (at least 37 of the 38 found): 70%. All 38: 55%. (The coarse grid step 0.5 missed Sprott case E in 3D; narrow windows may be lost.)
- P-E2: the pass finds more than 38 five-term families: 85% (the random run saw a family only if a draw landed in its window).
- P-E3: GE2 holds (speedup at least 2x): 70%. At least 5x: 40%.
- P-E4: GE3 holds: 75%.
- P-E5: wall time under 90 minutes on 6 workers: 55%.
- P-E6: at least one prediction is missed: 65%.

## Not claimed
That enumeration finds every 5-term chaotic flow: windows narrower than the grid, other starting points, and orbits needing a smaller step are not covered. Nothing about 6 terms: the same counts say a 6-term pass is not cheaper than sampling.

## Results (2026-10-28, written after the runs; no gate or prediction above was changed)

Files: `src/hyperchaos/keys.py`, `tests/test_keys.py`, `experiments/h2_enum5.py`, `h2_keys_compare.py`, `results/h2_enum5_controls.*`, `h2_keys_compare.*`, `h1_breakit_keys.log`. float64, Windows, CPU.

**Decision: the exhaustive 5-term pass is NOT enabled.** It failed gate GE3 before the full pass was run, so the 45-minute pass and the random-search baseline timing were not run (GE1, GE2 untested; no number for the speedup).
- **GE3 failed.** Through the pass's own worker (flip-reduced signs, 13-point invariant grid) Sprott A is found (as conservative), B, C, D are found, **E is not**. A follow-up on E alone: with the flip-reduced signs E is not found at any grid step tried (0.5, 0.25, 0.2, 0.125, 0.1), while the earlier all-signs control (`results/h2_fourterm_control_*`, step 0.1) found it with 36 systems. So the loss comes from keeping one sign representative per flip orbit, not only from the grid: the pipeline uses fixed start points that are not symmetric under sign flips, so a flipped copy can reach a different attractor basin. The reduction by flips is therefore not lossless here, and a full-sign, step-0.1 enumeration is, by my arithmetic, about 600 million systems (13,750 one-invariant classes x 61 x 32, plus 2,507 two-invariant x 61^2 x 32, plus 38 three-invariant x 61^3 x 32), about 12 times the roughly 5e7 five-term random draws it would replace. Not cheaper.
- **GE4 passed.** Keys unchanged by random rescaling, renaming and sign flip on Sprott A-E (3 tests), distinct for A-E; planted bug 8 (renamings only, no sign flips) is caught (bugs 1-8 all caught, 62 baseline tests).
- Keys are not a speedup and do not need the gate; they are enabled as an analysis tool.

**Predictions.** P-E1, P-E2, P-E3, P-E5: not tested (no full pass). P-E4 **missed** (75%: E not found). P-E6 hit.

**What the keys show (`results/h2_keys_compare.log`).** Night run, 4,634 hits after Amendment 5: **1,209 structure keys** (term-set plus signs, up to renaming and flips): 38 with 5 terms, 1,171 with 6; 1,093 term-sets when signs are ignored. Hits per structure key: median 1, maximum 217. 4,558 distinct (structure, parameter) pairs after rounding to 3 decimals. Sprott's cases A-S found by structure key in the night run: B, C, D, E, F, G, H, I, J, M, N (11 of 19; A is conservative and excluded by design; K, L, O, P, Q, R, S not found). In the 1e7 main run: B, C, D (3 of 19). **Correction to the night-run write-up:** I said none of the 472 earlier classes reappears; that was at the level of exact coefficient vectors. At the level of structure keys, 217 of the main run's 268 keys (81%) are found again in the night run.
Not compared: the 38 five-term families have not been checked against later literature beyond Sprott's table A-E; four of them are Sprott's B-E. No claim that any is new.
