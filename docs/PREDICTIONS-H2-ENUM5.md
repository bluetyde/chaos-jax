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
