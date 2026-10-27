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

## Results (2026-10-27)

Commands: `PYTHONPATH=src python experiments/kopp_chase.py results/kopp_chase.json` (parts table, starts, scan, dt, short; 145 rows, no errors) and `... results/kopp_long.json long`. Python 3.14.6, jax 0.12.0.1, RK4, QR every 0.1, 200 time units spin-up except the short runs. Machine shared with other jobs; wall times (about 17 s per T = 100000) are not results. Data: `results/kopp_chase.json`, `results/kopp_chase.log`, `results/kopp_long.json`.

| Part | Result |
|---|---|
| Table values of a, b = 0.55, start (1,1,1,1), T = 100000 | a = 0.15 quasi-periodic (0, 0, -0.075, -0.075; the paper says periodic). a = 0.55, 0.6, 0.62, 0.65, 0.8, 1.5 all **chaotic with one positive exponent**: second exponent 0.0000 to 0.0001, third about 0. Paper's second exponents: 0.0059, 0.0051, 0.0201, 0.0327, 0.0055, 0.0007. First exponent differs too: 0.1474 against 0.1385 at a = 0.62, 0.1446 against 0.0715 at a = 0.65. |
| a = 0.62 and 0.65, 14 start points each, T = 100000 | 13 of 14 starts at each a give the same spectrum within 0.001 (0.1474 +- 0.001, 0, 0, -0.7674 at a = 0.62). Largest second exponent anywhere: 0.0001. The 14th start is the origin (see below). |
| Scan a = 0.10 to 2.00, step 0.02, T = 20000 | 96 values: 72 chaotic, 24 quasi-periodic, **0 hyperchaotic**. Largest second exponent 0.0004. |
| dt 0.01, 0.005, 0.0025 at a = 0.62, 0.65 | Exponents change by at most 0.0006 (first) and 0.0001 (second). |
| Short runs from (1,1,1,1), no spin-up, T = 500 to 5000 | Second exponent between -0.0003 and 0.0004; the SE of the second exponent is 0.0064 at T = 500, 0.0040 at 1000, 0.0020 at 2000, 0.0005 at 5000. No short run gives 0.02. |
| a = 0.62, T = 1,000,000 | (0.1477, 0.0000, -0.0000, -0.7678). |

**Origin start (0,0,0,0):** the orbit is not on the attractor. `x1 = x2 = x4 = 0` stays zero and `x3` drifts to minus infinity linearly (max |x| 100200 at T = 100000), a non-smooth, unbounded orbit. My classifier labelled it "chaotic" with exponents (162, 0, 0, -98), but the sum check fails (error 65 to 71). That is a gap in `classify`: it needs a boundedness or sum-check condition. Not a Kopp result; not fixed here.

### Predictions against results

| Prediction | Result |
|---|---|
| P-K1 (85%) second exponent < 0.005 from every start at a = 0.62 | Hit, for every start on the attractor (the origin start is not on it). |
| P-K2 (40%) some a in the scan is hyperchaotic | No (0 of 96). |
| P-K3 (30%) a = 0.65 second exponent > 0.01 | No (0.0000 to 0.0001). |
| P-K4 (40%) a short run shows > 0.015 | No (largest 0.0004). |
| P-K5 (90%) dt changes < 0.002 | Hit (0.0006). |
| P-K6 (25%) more than one attractor at a = 0.62 | No (only the unbounded origin orbit, which is not an attractor). |

### Conclusion

**The published hyperchaos does not reproduce.** The transcription matches Eq. (1) of the paper. In every run this system has one positive exponent (about 0.147 at a = 0.62), two exponents at zero within error (the flow direction and a second one), and one large negative exponent. A second zero exponent on a 4D dissipative flow suggests a second conserved or marginal direction (the symmetry and the `x4 = -b * integral of x1` structure make it plausible), but I have not proved that. The paper's second exponent of 0.0201 is about 2.6 times its own flow-direction error (0.0076), and the short-run standard errors above (0.002 to 0.006) are of that size. The most likely explanation is a short or noisy run in the paper, but I did not run the paper's code or Mathematica, so that remains a hypothesis. The paper's first exponents also differ (0.1385 against 0.1474 at a = 0.62; 0.0715 against 0.1446 at a = 0.65), which is not explained by noise on the second exponent alone.

Consequence for the H0 table: Kopp 2025 does not count as a 7-term hyperchaotic flow. The "six papers with 7 terms" becomes five at most. Kopp and Samuilik 2025, Yu 2023 and PHYSCON 2015 are still not re-computed.
