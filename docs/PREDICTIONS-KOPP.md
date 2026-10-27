# Predictions: chasing Kopp 2025 (written before the runs, 2026-10-27)

Question: does the published hyperchaos of Kopp 2025 (TWMS J. Appl. Eng. Math. 15(11), 2584-2598) reproduce?

## Already known before the runs (not predictions)

- Transcription check done against the paper PDF (text extracted from `01a.pdf` on jaem.isikun.edu.tr): Eq. (1) is `x1' = a(-x1+x2)+x4, x2' = -x3 tanh x1, x3' = |x1|-1, x4' = -b x1`. This is what `kopp25` in `src/hyperchaos/systems.py` computes. Divergence is `-a`, as the paper says. Initial condition (1,1,1,1), `b = 0.55`, `a = 0.62`.
- The published exponents at a = 0.62 sum to -0.6202. Ours sum to -0.62 too. Both pass the divergence check, so the sum does not tell the two apart.
- The paper's Table 1 has a flow-direction exponent that should be exactly 0 but is printed as 0.0076 (a = 0.62), 0.0035 (a = 0.6), -0.0046 (a = 0.65). So the paper's own run error is about 0.005 to 0.008. The claimed second exponent 0.0201 is only about 2.6 times that error.

## What I will run

1. Paper's Table 1 values of a (0.15, 0.55, 0.6, 0.62, 0.65, 0.8, 1.5), b = 0.55, start (1,1,1,1), T = 100000.
2. a = 0.62 from many start points (the paper's, nearby, wide boxes, mirror images), T = 100000.
3. A scan of a from 0.10 to 2.00, step 0.02, T = 20000, for any window with a second exponent above the v2 threshold.
4. Step-size and integrator check at a = 0.62 and 0.65: dt 0.01, 0.005, 0.0025.
5. Short runs from (1,1,1,1) at a = 0.62 with T = 500, 1000, 2000, 5000 and no spin-up (the paper does not mention one), to see if a transient shows lambda2 near 0.02.
6. One long run, T = 1,000,000, a = 0.62.

## Predictions (probability that the statement is true)

- P-K1: at a = 0.62 the second exponent is below 0.005 at T = 100000 from every start point: 85%.
- P-K2: some a in the scan 0.10-2.00 labelled hyperchaotic by rule v2 (second exponent above max(0.005, 3 SE, 3 |z_err|)): 40%.
- P-K3: at a = 0.65 the second exponent at T = 100000 is above 0.01: 30%.
- P-K4: short runs (no spin-up, T <= 2000) show a second exponent above 0.015 for at least one T: 40%.
- P-K5: dt 0.005 and 0.0025 change the a = 0.62 exponents by less than 0.002: 90%.
- P-K6: more than one attractor at a = 0.62 (different exponent sets from different starts): 25%.

## What would change my mind

If any long run (T >= 100000) at some a from the paper's table gives lambda2 above 0.01 with rule v2 hyperchaotic, I report "reproduces at that a, not at a = 0.62". If none does, the report says "not reproduced; likely finite-run error in the paper", and I do not claim the paper is wrong beyond what these runs show.
