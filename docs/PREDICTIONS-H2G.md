# H2G predictions and gates: does a looser float32 stage-1 threshold recover the lost systems? (written 2026-10-27, before the run)

## Question
H2F (W2a) found that the float32 stage 1 keeps only 81.6% of the float64 stage-1 keepers (2658 of 3258 on seeds 1000-1059) and 464 of the 472 known hits. The keeper rule is: stage 1a passes (bounded and still moving), and in stage 1b the state stays below 1e4 and the one-vector exponent estimate is above 0.01. 40% of the lost keepers had a float64 exponent under 0.05, so a lower float32 threshold on the exponent might win them back, at the price of more survivors for the later stages. Nothing else changes: same draws (seeds 1000-1059), same stage 1a, same float32 code (`src/hyperchaos/screen32.py`), same 472 hits.

## Method
Rerun the float32 stage 1 on seeds 1000-1059 and on the 472 hits, storing for every stage-1a survivor the float32 exponent estimate and boundedness flag (the earlier run kept only the final keep or drop). The float64 keepers are recomputed on the same draws. Then, offline, sweep the float32 exponent threshold over 0.01 (current), 0.005, 0, -0.01, -0.02, -0.05, -0.1 and report for each: share of float64 keepers kept, hits kept (of 472), number of float32 keepers relative to the 3258 float64 keepers.

## Gates
- **GG1 (worth reopening the float32 GPU route).** At some threshold: at least 99% of the 472 hits kept (467), at least 97% of the float64 keepers kept, and at most 2.0 times as many float32 keepers as float64 keepers (3258). If no threshold passes, the float32 screen is closed for good and I say so.
- **GG2 (report).** Which of the 8 lost hits come back at which threshold, and which never do, with the reason (stage 1a, boundedness flag, or exponent).

## Predictions (probability the statement is true)
- P-G1: some threshold reaches at least 467 hits kept: 35%. All 472: 5%.
- P-G2: some threshold reaches at least 97% of the float64 keepers: 40%.
- P-G3: the threshold that reaches 99% of the hits (if any) needs more than 2.0 times as many float32 keepers as float64 keepers: 60%.
- P-G4: at least 3 of the 8 lost hits come back at threshold 0 or lower: 55%. At least 5 of 8: 15%. (Three of them have float64 exponents 0.013 to 0.027; the others have large exponents and were probably lost to a different behaviour of the float32 orbit.)
- P-G5: at least one prediction is missed: 75%.

## What would change my mind
If the lost hits with large exponents are lost by stage 1a or by the boundedness flag, no exponent threshold can bring them back, and the loss floor of the float32 screen is what the 8 lost hits say (1.7%). The decision whether 1.7% to 3% is acceptable stays with the owner.

## Results (2026-10-27, written after the run; no gate or prediction above was changed)

Files: `results/h2g_threshold.json`, `results/h2g_threshold.log`, `experiments/h2g_threshold.py`. Seeds 1000-1059 (1.2e6 draws), 6 workers, wall 469 s; float64 keepers 3258; the 472 known hits. Background load was low (desktop programs only).

| Float32 exponent threshold | float64 keepers kept (of 3258) | float32 keepers | hits kept (of 472) |
|---|---|---|---|
| 0.01 (current) | 2658 (81.6%) | 3270 (1.00x) | 464 |
| 0.005 | 2750 (84.4%) | 12,741 (3.91x) | 466 |
| 0 | 2820 (86.6%) | 107,562 (33.0x) | 467 |
| -0.01 | 2965 (91.0%) | 203,244 (62.4x) | 470 |
| -0.02 | 2976 (91.3%) | 204,177 (62.7x) | 470 |
| -0.05 | 2989 (91.7%) | 204,557 (62.8x) | 470 |
| -0.1 | 2997 (92.0%) | 205,055 (62.9x) | 470 |

**GG1 fails: no threshold gives at least 99% of the hits, at least 97% of the float64 keepers and at most 2.0 times as many float32 keepers.** Lowering the threshold to 0 recovers the hits only at the price of 33 times as many survivors, because every fixed point and quasi-periodic orbit (exponent about 0) then passes. The share of float64 keepers kept never rises above 92%.

**GG2, the 8 lost hits.** Three have a float32 exponent just under 0.01 and come back at 0.005 or 0 (hits 159, 402, 309; float64 exponents 0.016, 0.013, 0.027). Three more have a float32 exponent at or below zero and only come back at -0.01 (hits 51, 150, 426; float64 exponents 5.5, 0.099, 1.0): the float32 orbit settled somewhere else. Two never come back: hit 133 is dropped by float32 stage 1a, hit 168 gives a non-finite state in float32. So a threshold of 0 or lower recovers 3 to 6 of the 8, and 2 are lost whatever the threshold.

**Predictions judged.** P-G1 true (467 kept at threshold 0, but at 33x survivors; all 472 never). P-G2 **missed** (92% at best). P-G3 true. P-G4 true, and the second part (5 or more of 8 come back) true too at threshold -0.01 or lower (6 of 8); I had said 15%. P-G5 true (two misses).

**Conclusion.** The float32 stage 1 is closed as a screen for this search: either 1.7% of the hits are lost, or the survivor count rises by 4 to 63 times. The GPU float32 screening route stays cancelled unless the owner accepts a hit loss of about 1.7% at the current threshold.
