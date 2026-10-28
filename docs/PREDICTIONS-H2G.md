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
