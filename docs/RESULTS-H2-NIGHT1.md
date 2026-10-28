# H2 night run 1: 1e8 uniform draws (run 2026-10-27 22:20 to 2026-10-28 05:30)

Predictions were given in chat before the run, not committed in a file (miss on process: the owner asked for them beside the results, so this file records them as stated): about 4,700 hits, 7.5-8 h on 8 workers, about 2.3% hit loss from the boundedness and float32 filters, no hung phase.

Command: `experiments/h2_night.py night1 8 20 250 10000` via `run_night1.bat`. Uniform sampler (no weights), registered grid, k = 5 or 6 terms (50/50), seeds 10000-14999, float32 pass on. Files: `results/h2_night1_b000..019.json`, `results/h2_night1_summary.json`, `results/h2_night1.log`, `results/h2_night1_analysis.log`, `experiments/h2_night_analyze.py`.

## Result
| | Predicted | Measured |
|---|---|---|
| Wall time | 7.5-8 h | 7.2 h (25,837 s of block time; log 22:20 to 05:30) |
| Confirmed hits | about 4,700 | 4,673 (1 per 21,400 draws) |
| Hung phase | none | none, exit code 0 |
| Phase split (screen / float32 / float64) | 37/17/47 | 40/17/43 |

Funnel: stage 1a 23,268,280, stage 1b 270,033, boundedness 110,356, float32 13,123, float64 stage 2 5,821, confirmed 4,673.

- All 4,673 hits are distinct under the 48-element symmetry group (no duplicates). 3,570 have k = 6, 1,103 have k = 5. No k = 4 systems exist in this run: H2 draws k = 5 or 6 only.
- None of the 472 classes from the 1e7 main run reappears tonight. So the space is far from exhausted: 1e8 draws did not saturate the class count.
- The 2.3% hit loss is not measured by this run (it records no dropped systems); it comes from the earlier recall tests on seeds 1000-1059.

## Flags (not resolved)
- 13 hits have a first exponent above 10 (44 above 5), and 5 have a second exponent above 0.01. A 3-D flow cannot have two positive exponents, so the five with lambda2 > 0.01 are estimate noise (standard errors of the same size). The 13 with lambda1 above 10 (most with lambda2 near zero) look stiff, and RK4 at dt 0.01 may be unreliable for them. The classifier's sum check passed, but I have not re-integrated these at a smaller step. Treat them as unverified until that is done.
- Hit counts are raw confirmations at the classifier rule v2, not a published-quality catalogue.
