# Predictions: chasing Kopp and Samuilik 2025 (written 2026-10-27, before any run)

Question: does the published hyperchaos of Kopp and Samuilik, Mathematics 13(17), 2838 (2025), reproduce? Same method as `docs/PREDICTIONS-KOPP.md`.

## Known before the runs (not predictions)

- The equations are copied from my earlier partial read (`LITERATURE.md`, system 6, Eq. 3): `x1' = a x2 - (a+alpha) x1 - beta |x4| x1`, `x2' = -x1 x3`, `x3' = x1 x2 - b`, `x4' = x1`, with alpha = 1, beta = 0.1525, b = 90. I could not re-check them against the PDF today: the MDPI site answers HTTP 403 to my fetch tool (page, XML and PDF). A transcription error is therefore not ruled out, unlike for Kopp 2025.
- **I do not know the paper's initial conditions or integrator settings** (that part of the paper is not in my notes and I cannot fetch it). The runs below use many starts instead.
- Published exponents: a = 3: (1.0087, 0.0328, -0.0049, -7.7900); a = 5: (0.8859, 0.0556, -0.0071, -10.8456). As in Kopp 2025, the "zero" exponent is printed as -0.0049 and -0.0071, so the paper's own run error is about 0.005 to 0.007; the claimed second exponents are 0.0328 and 0.0556, about 7 and 8 times that.
- Divergence is `-(a+alpha) - beta |x4|` (only the first equation contributes). The published sums are -6.7534 (a = 3) and -10.8456+0.8859+0.0556-0.0071 = -9.9112 (a = 5). So the published numbers imply a time average of `|x4|` of (6.7534-4)/0.1525 = 18.05 at a = 3 and (9.9112-6)/0.1525 = 25.65 at a = 5. My runs report the average of `|x4|` as well, so this is a check that I am on the same attractor as the paper, independent of the second exponent.

## What I will run (`experiments/kopp_samuilik.py`)

1. a = 3 and a = 5, T = 100000, dt = 0.01, spin-up 500, from 12 start points: (1,1,1,1), (0.1,0.1,0.1,0.1), (1,0,0,0), (0,0,1,0), (-1,1,0,1) and 7 random draws in boxes of half-width 1 and 10 (seed fixed).
2. dt 0.01, 0.005, 0.0025 at a = 3 from the first start that gives a bounded orbit.
3. Short runs (T = 500, 1000, 2000, 5000, no spin-up) at a = 3 from (1,1,1,1).
4. A scan of a from 1.0 to 8.0 in steps of 0.25, T = 20000, for any window labelled hyperchaotic by rule v2.

## Predictions (probability that the statement is true)

- P-S1: at a = 3, at least one start gives a bounded attractor with a first exponent within 0.05 of 1.0087: 65%.
- P-S2: on that attractor the implied average of `|x4|` is within 5% of 18.05: 55%.
- P-S3: at a = 3 the second exponent is below 0.005 at T = 100000 on every start that reaches that attractor (not reproduced): 60%.
- P-S4: some start at a = 3 or 5 gives rule-v2 hyperchaos (second exponent above max(0.005, 3 SE, 3 |z_err|)): 30%.
- P-S5: the scan finds at least one a with rule-v2 hyperchaos: 30%.
- P-S6: more than one attractor at a = 3 (different exponent sets from different starts, not counting unbounded orbits): 40% (the abstract mentions multistability).
- P-S7: dt changes the exponents by less than 0.005: 80%.
- P-S8: the attractor is not found at all at a = 3 (every start diverges or my equations are wrong): 15%.

## What would change my mind

If P-S2 fails badly (average `|x4|` far from 18), I will say the first thing to suspect is my transcription of the equations, not the paper, and I will not call the result a failure to reproduce. If any start gives rule-v2 hyperchaos with the second exponent above 0.02 at T >= 100000, I report "reproduces from that start, at that a".

## Results (2026-10-27)

Command: `PYTHONPATH=src python experiments/kopp_samuilik.py results/kopp_samuilik.json` (12 starts at a = 3 and a = 5, dt check, short runs, scan a = 1 to 8; 61 rows, no errors; log in `results/kopp_samuilik.log`). Python 3.14.6, jax 0.12.0.1, RK4, QR every 0.1, spin-up 500, T = 100000 unless stated.

**Summary: the system as I transcribed it is not the paper's system, so this run says nothing about whether the paper's hyperchaos is real.**

| Check | Paper | My runs |
|---|---|---|
| a = 3 exponents | (1.0087, 0.0328, -0.0049, -7.79) | largest exponent 0.12 to 0.35, smallest -31 to -41; second exponent 0.003 to 0.006 |
| implied average of `|x4|` at a = 3 | 18.05 | 180 to 250 |
| a = 5 exponents | (0.8859, 0.0556, -0.0071, -10.85) | largest 0.01 to 0.36, smallest -29 to -55 |
| implied average of `|x4|` at a = 5 | 25.65 | 150 to 324 |
| scale of the state | not known | largest |x| about 300 to 1500 |

The implied average of `|x4|` is off by a factor of about ten, and the exponents are far from the published ones (the smallest is -31 to -41 against -7.8). By the rule written before the runs, this means the first suspect is my transcription of the equations or parameters (the equations come from a partial read, I cannot fetch the PDF, and `b = 90` in particular gives states of order a thousand), not the paper. I do not call this a failure to reproduce.

What the runs do show, for this transcribed system only:
- No start at a = 3 or 5, and no a in the scan from 1 to 8, is labelled hyperchaotic by rule v2. The largest second exponent is 0.026 (a = 5, start 1) with a standard error of 0.016, and that run is labelled "inconsistent".
- Most runs are labelled "inconsistent": the exponent sum does not match the average divergence within the G1 tolerance, so these exponents are not trustworthy either. The orbit also changes label with the step size (a = 3: quasi-periodic at dt 0.01 and 0.0025, chaotic at 0.005; first exponent 0.025, 0.231, 0.010), which points at a non-smooth, sensitive system (the `|x4|` term) or a poorly resolved one.
- The start (0, 0, 1, 0) diverges (x1 = x2 = x4 = 0 is invariant), as the origin does for Kopp 2025.

### Predictions against results

| Prediction | Result |
|---|---|
| P-S1 (65%) first exponent within 0.05 of 1.0087 | **Miss.** |
| P-S2 (55%) implied `|x4|` within 5% of 18.05 | **Miss**, by a factor of ten. |
| P-S3 (60%) second exponent below 0.005 on every start that reaches the attractor | **Miss on the strict wording**: three of the a = 3 starts give 0.0051, 0.0056, 0.0059 (all within about one to two standard errors of zero). |
| P-S4 (30%) some start gives rule-v2 hyperchaos | No. |
| P-S5 (30%) the scan finds hyperchaos | No. |
| P-S6 (40%) more than one attractor at a = 3 | **Unresolved**: exponent sets vary a lot between starts, but the "inconsistent" labels and the dt dependence mean I cannot tell attractors from poor convergence. |
| P-S7 (80%) dt changes exponents by less than 0.005 | **Miss** (first exponent 0.010 to 0.231). |
| P-S8 (15%) the paper's attractor is not found / my equations are wrong | **Hit in effect**: nothing like the published spectrum appears. |

### What is needed to settle it

The paper's Eq. (3) with its parameter values and initial conditions, as printed. Either paste Section 2 and the exponent table, or download the PDF from MDPI (https://www.mdpi.com/2227-7390/13/17/2838) into the folder `refs-owner` here (not committed, per `NEEDED-FROM-OWNER.md`). Then re-run with the paper's start and settings before any statement about the paper.
