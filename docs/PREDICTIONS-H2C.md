# H2C predictions and gates: a long-horizon boundedness filter before stage 2 (written 2026-10-27, after a calibration look, before the filter was wired in or any gate run)

## Why and what was looked at first

H2B showed that a short exponent test does not remove the stage-1b survivors that later fail, because they fail by escaping or drifting after stage 1b's 600-time-unit horizon. This stage asks: does a cheap state-only run over the full stage-2 horizon (no tangent frame, no QR, batched like stage 1) catch them?

**Calibration data (disclosed; not used for any gate).** Before writing this I ran `experiments/h2c_explore.py` on seeds 2000-2005 (new seeds, not in the main run): 343 stage-1b survivors, each with its H1 stage-2 label and the state-only statistics (`results/h2c_explore.json`). Findings: 143 of 343 survivors are labelled "diverges" and 138 of those exceed |x| = 1e4 in the state-only run (first excursion between t = 600 and 2139, median 737); of 112 "inconsistent", 30 exceed 1e4; of 60 quasi-periodic (with or without the conservative flag), 28 exceed; 8 more (7 fixed points and 1 inconsistent) are stuck (late range under 1e-3). In total 196 escapes and 8 stuck of 336 failing survivors. The 7 survivors that stage 2 labels chaotic and not conservative (plus 1 chaotic and conservative) were never flagged (largest |x| 144, smallest late range 0.95). The state-only run costs about 1 s per chunk, stage 2 about 75 s per chunk (6 workers in parallel). Because the rule and the thresholds below were chosen after seeing this, the predictions that depend on it (P-C3, P-C4) are not independent of it; the gates are run on other data.

## Definition fixed now

State-only RK4 from the fixed start, dt = 0.02, T = 5000 (250,000 steps), for every stage-1b survivor, batched with `vmap`. A system is **dropped** if the state leaves the finite region |x| <= 1e4 at any step (or becomes non-finite), or if the range of any coordinate over the last tenth of the run is below 1e-3 (stuck). Otherwise it goes to stage 2 as before. Stage 2 and confirmation are unchanged. The limit 1e4 is the one stage 1a and 1b already use; the stuck limit 1e-3 is the one stage 1a uses.

## Gates

- **GA (recall).** All 472 confirmed hits of `results/h2_main.jsonl` through the filter alone. Pass: at least 95% (449) kept. Every dropped hit is listed.
- **GB (same draws, same hits).** The filtered pipeline on seeds 1000-1059 (1.2e6 draws, the same as the old run). Pass: it recovers at least 95% of the old hits for those seeds and finds none that the old run did not.
- **GC (speed).** Old and new pipelines run back to back on seeds 1000-1005 under the same conditions (6 workers, same machine). Pass: the new one takes at most two thirds of the old worker-seconds (1.5x or better). I will say what else was running.
- **GD (planted bugs).** Tests for the filter fail when it is broken in two ways: escape test inverted; stuck limit set to zero (sinks no longer dropped).
- **GE (report).** Survivors reaching stage 2 as a share of stage-1b survivors, cost of the filter per chunk, speedup, hits lost.

## Predictions (probability that the statement is true)

- P-C1: GA passes (95% or more kept): 97%. All 472 kept: 85%.
- P-C2: the hit set on seeds 1000-1059 is identical to the old one: 80%.
- P-C3: at most 50% of the stage-1b survivors reach stage 2: 85%. At most 40%: 55%. (Calibration gave 41%.)
- P-C4: GC passes (1.5x or better): 80%. 2x or better: 45%.
- P-C5: the filter costs at most 3 s per 20000-draw chunk: 85%.
- P-C6: GD passes on the first run: 85%.
- P-C7: at least one prediction here is missed: 85%.

## What would change my mind

If GA fails because real hits leave |x| = 1e4 transiently, the limit is too tight for the registered search and a larger limit has to be calibrated; I would not loosen it without new data. If the speedup is under 1.5x with full recall, the remaining cost is in the 60% of failures the filter cannot see ("inconsistent", conservative quasi-periodic), and the next question is what distinguishes them from true hits cheaply.

## Results (2026-10-27, written after the runs; no gate or prediction above was changed)

Files: `results/h2c_recall.json`, `results/h2_h2c_new.jsonl/.log` (filtered pipeline, `H2_FILTER=bound`, seeds 1000-1059), `results/h2_h2c_old6.jsonl/.log` (old pipeline, seeds 1000-1005, run right after the new one), `results/h2c_explore.json` (calibration, seeds 2000-2005). The machine was about 28% busy with other jobs before and after the runs, so only the same-conditions comparison counts.

**The boundedness filter gives a 1.7x speedup with the same hits.**

| Gate | Result |
|---|---|
| GA recall on the 472 known hits | **pass**: 470 of 472 kept (99.6%). The two lost: hit 100 (k = 6, exponents 0.241, 0.0001, -0.422) leaves 1e4 at t = 988 in the dt = 0.02 state-only run although the dt = 0.01 evaluator keeps it bounded for 20000 time units; hit 215 (k = 6, exponents 7.35, 0.0, -1.06, a very large first exponent) leaves 1e4 at t = 2381. |
| GB same draws, same hits | **pass**: 63 old hits, 63 new, identical, none new, none lost |
| GC speed (1.5x or better) | **pass**: seeds 1000-1005 back to back, new 302 worker-seconds against 522 for the old pipeline, 1.73x. Over all 60 seeds the new run took 305 worker-seconds per 1e5 draws against 502 for the old run's file (1.65x, but the old figure was measured under other load) |
| GD planted bugs | **pass**: both new bugs caught (8 of 8 in the script, tree clean after) |

Share of stage-1b survivors reaching stage 2: 1366 of 3258 (41.9%). Stage 2 still sees the same 76 chaotic ones and 63 confirmed.

**Predictions judged.** P-C1: 95% or more kept true, **all 472 kept missed** (470). P-C2 true. P-C3: at most 50% true, **at most 40% missed** (41.9%; the calibration had said 41%). P-C4: 1.5x or better true, **2x or better missed** (1.73x). P-C5 true on the exploration timings (0.8 to 1.5 s per chunk, one process per seed; the search log did not record it separately). P-C6 true. P-C7 true (three misses).

**What remains.** About 58% of the failing survivors still reach stage 2: 81 of the "inconsistent", 23 conservative quasi-periodic, 13 fixed points and 9 quasi-periodic ones in the calibration sample did not leave 1e4 or get stuck. The next cheap test would have to tell them from true hits without the tangent frame. I have not looked at what distinguishes them. A 1e9-draw run at the new speed would take roughly 6 days of this machine's CPU instead of 10 (my estimate from 1.7x, not measured), which is still long.

**Caveat on the two lost hits.** Both are hits the old pipeline confirmed at T = 20000 from two starts at dt = 0.01. The filter uses dt = 0.02, so a 0.4% loss of true hits is a real cost of the cheaper step; if it matters, the filter's step can be set to 0.01 (twice the 1 s cost).
