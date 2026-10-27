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
