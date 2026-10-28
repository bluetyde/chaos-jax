# H2 four-term fine pass on rank-3 supports: predictions and gates (written 2026-10-28, before the run)

Follow-up to `docs/PREDICTIONS-H2-FOURTERM.md`. Result there: no dissipative 4-term chaotic flow, but the 4,881 rank-3 supports were scanned only at step 0.5 in their one scale invariant, and that grid missed Sprott case E (narrow chaotic window).

## Method
Same pipeline and code as the four-term search (`experiments/h2_fourterm.py` worker), rank-3 supports only, invariant grid ln t = -3.0 to +3.0 in steps of 0.1 (61 values; the control that found all of B-E used this grid), 16 sign patterns: 4,881 x 61 x 16 = 4,764,000 systems. Runs in 489 chunks of 10 supports, 6 workers, one result file per chunk (resumable); a stall of 30 minutes with no chunk finishing kills the workers and stops the run. Expected wall time about 100 to 120 minutes (the coarse pass did 1.0M rank-3 systems in about 20 minutes; 4.7 times as many).
Conservative candidates are screened for a quadratic first integral a^2 - b^2 (derivative at 200 random points), as in the four-term check.

## Gates
- **GN1 (completeness).** All 489 chunk files exist and the counts add to 4,764,000 systems.
- **GN2 (hit handling).** Any dissipative hit is re-run at dt 0.005 and 0.0025 from both starts, with the exponent sum checked negative, before it is reported as anything but a candidate.
- **GN3 (coarse-grid agreement).** Every system that the coarse pass kept through stage 2 (none: zero hits) stays consistent: the fine pass finds zero where the coarse pass found zero is not evidence; the check is that the fine grid contains the coarse grid points (ln t multiples of 0.5 are multiples of 0.1), so the fine counts at those points must reproduce the coarse counts (0 hits, 5 conservative candidates). If the fine run does not re-find the 5 conservative candidates, something differs between the runs.

## Predictions (probability the statement is true)
- P-N1: no dissipative chaotic hit: 70%. At least one: 30%. At least five: 5%.
- P-N2: GN3 holds (the same 5 conservative candidates appear among the fine-run conservative candidates): 85%.
- P-N3: every fine-run conservative candidate has a first integral: 85%.
- P-N4: wall time between 90 and 150 minutes: 65%.
- P-N5: at least one prediction is missed: 55%.

## Not claimed
Anything about the 273 rank-2 supports (two invariants; a fine pass would need about 16 million systems and is not run here), or about invariants outside ln t in [-3, 3].
