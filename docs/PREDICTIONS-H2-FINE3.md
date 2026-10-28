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

## Results (2026-10-28, written after the run; no gate or prediction above was changed)

Files: `experiments/h2_fine3.py`, `h2_fine3_analyze.py`, `results/h2_fine3/` (489 chunk files), `results/h2_fine3_summary.json`, `results/h2_fine3.log`, `results/h2_fine3_analysis.{log,json}`. float64, 6 workers, Windows, CPU. Wall time 2,708 s (45 min).

**Headline: no dissipative chaotic 4-term flow among the rank-3 supports at invariant step 0.1.** 4,763,856 systems screened; stage 1a survivors 1,768,902; stage 1b 32,237; boundedness 2,909; stage 2 (dissipative, chaotic, T = 5000): **0**. 35 conservative candidates, on 2 distinct supports.
**GN1** passes (489 of 489 chunk files; the exact count is 4,881 x 61 x 16 = 4,763,856, the 4,764,000 in the method section was rounded). **GN2** not triggered (no dissipative hit). **GN3**: the prediction was mis-specified. Only the 3 rank-3 candidates of the coarse pass can reappear (the other 2 of the 5 are rank 4, outside this pass); all 3 of 3 reappear in the fine pass, same coefficients. The check holds in its corrected form.
**All 35 conservative candidates have a quadratic first integral** (a^2 - b^2 of two coordinates, derivative below 1e-9 at 200 random points each): none can be chaotic.

**Predictions.** P-N1 hit (no dissipative hit). P-N2 **missed as worded** ("the same 5"): the correct count is 3 and it holds. P-N3 hit (35 of 35). P-N4 **missed**: 45 minutes, not 90 to 150 (my per-system cost estimate was about 2.3 times too high; the early chunks and most 4-term systems are cheap). P-N5 hit.

**What this changes.** The rank-3 half of the earlier caveat is closed: for rank-3 supports the grid is now step 0.1 in the invariant over ln t in [-3, 3], the grid on which the controls B-E were all found. Left open: the 273 rank-2 supports (two invariants, coarse grid 0.5 only) and anything outside ln t in [-3, 3] or outside the START and START2 starting points. From this run's rate (4.76M systems in 45 min) the rank-2 fine pass (273 x 61 x 61 x 16 = 16,253,808 systems) would take about 2.6 hours, not "many hours".
