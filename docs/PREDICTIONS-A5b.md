# Task A5b predictions and gates: hyperbolicity test for Lorenz-63

Written and committed before any A5b code ran. Gates follow `TODO.md` (task A5b) and are made exact here.
Do not edit a gate or prediction after results. Add a dated Amendment instead.

## Question

My three shadowing codes agree with each other (rho 1.017, sigma 0.134) and differ from the finite-difference oracle (rho 1.003, sigma 0.149). Can the geometry of Lorenz-63 at rho = 28 (how close the unstable and stable directions come to each other) explain why shadowing and finite differences differ?

## What I read for this task (logged) and what it changes

- `arxiv-1806.02077` sections 5.3, 5.4 and the conclusion: same observable, parameters and finite-difference step as mine. Shadowing-type methods and unstable-periodic-orbit sensitivities converge to about 1.017 for rho. Finite differences are lower. It ties the bias to the lack of uniform hyperbolicity. It also shows, in figure 10, that the bias varies with rho: in its words the bias is "as small as the standard error bars" near rho = 33, and larger at low rho.
- `arxiv-1212.3961` (Ginelli et al.) sections 4.1 and 4.2: the algorithm I implement. Equations 30 to 32 checked on the page image.
- `arxiv-1105.5228` (Kuptsov, Parlitz): the hyperbolicity test by angles between the expanding and contracting subspaces. If their distribution is well separated from zero the chaos is hyperbolic. If it approaches zero, hyperbolicity is violated.
- `arxiv-1202.1571` (Morriss): in a hard-disk system exact tangencies do not occur for a generic trajectory. It says nothing about Lorenz.
- **`arxiv-2009.00595` (Ni): a second explanation that this task cannot test.** For hyperbolic systems the linear response is the sum of a **shadowing contribution** and an **unstable contribution**. NILSS and LSS compute only the shadowing contribution. Ni states that the shadowing part alone is a reasonable approximation when the unstable dimension u is much smaller than the dimension M. For Lorenz-63 u = 1 and M = 3. So the gap could be a missing unstable contribution, which would exist even if Lorenz-63 were uniformly hyperbolic. Computing the unstable contribution needs Ni's second-order tangent algorithm (for maps in the paper I have). I do not implement it here.

## What this test can and cannot do (fixed now)

- It measures angles between the covariant Lyapunov vectors along long orbits and relates them to the shadowing contributions and to the gap.
- It **can** show whether near-tangencies occur, and whether the shadowing estimate behaves differently where they occur.
- It **cannot** say which of shadowing and finite differences is closer to the true derivative.
- It **cannot** separate "lack of hyperbolicity" from "missing unstable contribution". Either could give the same gap.
- An association across a handful of rho values is not a cause.

## Method (fixed now)

Lorenz-63 (sigma, beta) = (10, 8/3), RK4. Covariant Lyapunov vectors (CLVs) by the dynamical algorithm of Ginelli et al., written by me: forward evolution of an orthonormal tangent basis with QR every block (storing Q_n and R_n), a backward transient, then the backward recursion C_n = R_{n+1}^-1 C_{n+1} with unit columns (the QR gauge has a positive diagonal). Block length 0.1 time units. Forward transient 50 time units (Gram-Schmidt convergence), backward transient 100 time units, both discarded.
Vectors: zeta_1 (unstable, exponent about +0.90), zeta_2 (neutral, parallel to the flow f), zeta_3 (stable, exponent about -14.57).
Angles (lines, in [0, pi/2]):
- theta1 = angle(zeta_1, zeta_3), unstable against stable;
- theta2 = angle between zeta_3 and the plane spanned by zeta_1 and f (stable against centre-unstable; the splitting that matters for a flow);
- theta3 = angle(zeta_1, f), unstable against flow.
Also |f| along the orbit (small near the equilibrium at the origin).

## Controls first (all before any Lorenz angle is read)

- **C-B1 linear flow with known CLVs:** dx/dt = A x with A non-normal, upper triangular, eigenvalues 1, -1, -3. The CLVs are the eigenvectors. The algorithm must reproduce them to an angle below 1e-8 in the interior of the orbit. Unit test.
- **C-B2 neutral vector:** for Lorenz at rho = 28, zeta_2 is parallel to f. The angle must be below 1e-6 for all interior samples. Unit test.
- **C-B3 exponents:** the mean growth rates along zeta_1, zeta_2, zeta_3 (from the R factors) must match 0.904 (within 0.02), 0 (within 0.02) and -14.57 (within 0.1). They are compared with the Phase 1 values.
- **C-B4 independence of the backward start:** two different random starting matrices give interior CLVs that agree to an angle below 1e-8.
- **C-B5 covariance without the QR bookkeeping:** propagating zeta_n over one block with the true tangent flow (jvp) gives zeta_{n+1} to an angle below 1e-8.

## Decision rules (fixed now)

**D1, are the angles bounded away from zero at rho = 28?** From two independent orbits of T = 10000 (100000 samples each after the transients).
- For each of theta1, theta2, theta3, report the minimum, the quantiles (0.1%, 1%, 5%, 50%) and the empirical P(theta < 0.01), P(theta < 0.05), P(theta < 0.1).
- "Bounded away from zero" for an angle if its minimum over both orbits is at least 0.05 rad and P(theta < 0.1) is below 1e-3.
- "Near-tangency recurrences" if P(theta < 0.05) is at least 1e-3.
- Between the two: "in between". The verdict is given for each of the three angles. The two orbits must agree on it.

**D2, do the segments that are close to a tangency carry the gap?** At rho = 28, with my NILSS per-segment contributions (A5 settings: 32 orbits, T = 1000, segments of 0.5 time units, the same orbits for the CLVs). Per segment: the contribution to d<z>/d(rho) per unit time, c_i; the smallest of each criterion over the segment's CLV samples; the norm of the shadowing direction at the segment end.
- Use only segments between t = 50 and t = 900 of each orbit (after the forward transient, before the backward transient).
- For each criterion k in {theta1, theta2, theta3, |f|}: remove the 5% of segments with the smallest criterion. Compute the estimate from the other segments (the mean of c_i) and its shift Delta_k against the mean of all segments. The null is 1000 random removals of the same number of segments. The gap is the difference between my NILSS (A5, 1.01701) and the oracle (1.00317), 0.01384.
- Criterion k **explains the gap** if Delta_k lies outside the 95% interval of the null AND moves the estimate toward the oracle by at least half the gap (0.0069). Otherwise it does not.
- Also reported: Spearman correlation between log of each criterion and |c_i|, and between log theta and the log shadowing-direction norm. (The nilss-jax documentation suggests the norm of the shadowing direction is of order 1/sin(angle).)

**D3, does the gap follow the angle statistics across rho?** For rho in {26, 28, 30, 33, 36, 40}:
- gap(rho) = (my NILSS - oracle) / oracle. NILSS: 32 orbits, T = 1000, A5 settings. Oracle: finite differences as in A4 (steps 1.0 and 0.5, start points doubled from 4000 until the SE is below 5%). The oracle is "converged" if the two steps agree within 2 combined SE. The gap uses the h = 0.5 slope, and the difference to the h = 1.0 slope is reported as its systematic uncertainty.
- Angle statistics from one orbit of T = 10000 per rho: P(theta1 < 0.1), P(theta2 < 0.1), P(theta3 < 0.1), and the 1% quantile of |f|.
- Report the Spearman correlation between gap and each statistic (6 points). I call it an **association** if its magnitude is at least 0.8. With 6 points a correlation of 0.8 has a two-sided p of about 0.1.
- Also compared with 1806.02077: the gap should be smallest near rho = 33 and larger at low rho.

**Literature check on sigma:** within the library only (the full text files in `reference/text`). Search for any sensitivity of an average to sigma. No web search.

## Predictions (not gates)

- **P-B1 controls:** C-B1 to C-B5 pass at the first run. Any repair counts as a miss.
- **P-B2 D1:** theta1 and theta2 are bounded away from zero (minimum above 0.05 rad), at rho = 28. For theta1 I expect the minimum above 0.3 rad, because at the equilibrium the unstable and strongest stable eigenvectors in the x-y plane are about 63 degrees apart (from the Jacobian at the origin). **theta3 (unstable against flow) shows near-tangency recurrences** with P(theta3 < 0.05) between 0.5% and 10%, because the flow direction aligns with the unstable direction when a trajectory leaves the equilibrium region. So the pair zeta_1 and zeta_2 merge, which matches the singular-hyperbolic picture of Lorenz-63 (an equilibrium inside the attractor).
- **P-B3 D2:** no criterion explains the gap. I give this 70% (30% that one criterion, most likely |f|, explains at least half of it). The reason is that the contributions come from a thin sum of many segments, and the gap is 1.4% of the value.
- **P-B4 D2 correlations:** the correlation between log |f|_min and |c_i| is negative (stronger contributions near the equilibrium), with Spearman between -0.1 and -0.5.
- **P-B5 D3:** the gap changes with rho. It is small (below 0.5%) at rho = 33 and larger (above 1.5%) at 26 and 28. The correlation of the gap with the theta3 statistic is in magnitude below 0.8 (no association). I give 60% to "no association".
- **P-B6 sigma literature:** no paper in the library reports a sensitivity to sigma on an average of z. Only the nilss-jax README does (not in the library).
- **P-B7 compute:** about 10 minutes in all. The CLV forward runs are the slowest part. I time one before the main run.

## Amendment, 2026-10-27: first run of the CLV controls

The first run of the controls gave: C-B3, C-B4 and C-B5 pass. **C-B1 and C-B2 failed.** The CLV algorithm was not at fault in either case. I changed no gate.

- **C-B2 (neutral CLV against the flow direction, gate 1e-6).** The maximum angle was 4.5e-6 at dt = 0.01. It falls to 2.4e-7 at dt = 0.005 and 2.6e-8 at dt = 0.0025, which is fourth order in dt. The cause is that the discrete RK4 tangent map has a neutral direction that differs from the continuous flow direction at O(dt^4). I had not stated a dt for the CLV runs. **All A5b CLV runs use dt = 0.005**, the same as my NILSS (A5), where C-B2 passes (2.4e-7). The test now runs at dt = 0.005. The gate stays at 1e-6.
- **C-B1 (linear flow with known eigenvectors, gate 1e-8).** My test orbit was too short: about 5 time units of forward history and about 5 of backward convergence, while the exponent gaps of 2 need about 10 each for 1e-8. The test was badly designed. I now use 200 time units and check the interior from t = 20 to t = 180. The gate stays at 1e-8.
- **Angle function.** At the next control run C-B5 read 1.49e-8, above its gate of 1e-8. That value is the floor of `arccos` for a cosine of 1 - 1.1e-16, so my angle function could not resolve small angles. Earlier passes were lucky exact zeros. I replaced it with atan2 of the perpendicular and parallel parts, which is accurate at machine precision. All five controls then pass: C-B1 and C-B4 to 1e-8, C-B2 at dt = 0.005, C-B3 with exponents 0.9085, -0.0002 and -14.575, and C-B5.

## Results (added after the runs, 2026-10-27)

`python -m pytest -q` gives the line quoted in `HANDOFF.md`. Commands: `python experiments/phaseA5b_clv.py angles | link | scan` (9 s, 54 s, 31 s). Output in `results/phaseA5b_*.json` and `results/phaseA5b_link.npz`. The raw angle arrays are regenerated by `angles`.
Controls C-B1 to C-B5 pass after the repairs recorded in the Amendment. All CLV runs use dt = 0.005 and blocks of 0.1 time units.

### D1: angles at rho = 28 (two orbits, T = 10000, 98500 samples each after the transients)

| Angle | Orbit 0: min, 1% quantile, P(<0.05), P(<0.1) | Orbit 1: min, 1% quantile, P(<0.05), P(<0.1) | Verdict |
|---|---|---|---|
| theta1, unstable against stable | 0.1231, 0.398, 0, 0 | 0.1339, 0.399, 0, 0 | **bounded away from zero** |
| theta2, stable against centre-unstable plane | 0.1203, 0.385, 0, 0 | 0.1306, 0.385, 0, 0 | **bounded away from zero** |
| theta3, unstable against flow | 0.0071, 0.048, 0.0112, 0.0556 | 0.0083, 0.049, 0.0105, 0.0562 | **near-tangency recurrences** |

Both orbits agree on every verdict. The smallest |f| seen is 7.65 and 9.72 (median 77), so the orbits did not pass very close to the equilibrium at the origin at a sampling step of 0.1.

### D2: do the segments near a tangency carry the gap? (rho = 28, 54400 segments from 32 orbits, estimate from all of them 1.01691)

| Criterion (lowest 5% of segments removed) | Shift of the estimate | Outside the random-removal 95% interval [-0.00044, +0.00041]? | Explains the gap (needs a shift of at most -0.0069)? |
|---|---|---|---|
| theta1 | -0.00006 | no | no |
| theta2 | -0.0000008 | no | no |
| theta3 | +0.0000008 | no | no |
| smallest \|f\| | -0.00020 | no | no |

The Spearman correlations of the log criterion with the per-segment contribution and with the size of the shadowing direction are all below 0.009 in magnitude. The size of the shadowing direction does not follow 1/sin(angle) here.

### D3: the gap against rho (own NILSS minus finite differences at h = 0.5, as a fraction of the finite difference)

| rho | own NILSS | finite differences (h = 0.5; h = 1) | gap | oracle converged | P(theta1<0.1) | P(theta3<0.1) |
|---|---|---|---|---|---|---|
| 26 | 1.0226 | 0.9789 +- 0.0029; 0.9637 | +4.47% | **no** | 0 | 0.0651 |
| 28 | 1.0169 | 1.0035 +- 0.0021; 1.0044 | +1.34% | yes | 0 | 0.0565 |
| 30 | 1.0129 | 1.0042 +- 0.0020; 1.0075 | +0.87% | yes | 0 | 0.0485 |
| 33 | 1.0068 | 1.0108 +- 0.0021; 1.0076 | -0.40% | yes | 0.0006 | 0.0413 |
| 36 | 1.0073 | 0.9974 +- 0.0020; 0.9962 | +1.00% | yes | 0.0016 | 0.0371 |
| 40 | 0.9715 | 0.9815 +- 0.0017; 0.9839 | -1.02% | yes | 0.0043 | 0.0342 |

Spearman correlation of the signed gap with P(theta1<0.1) and P(theta2<0.1): -0.81. With P(theta3<0.1): +0.83. With the 1% quantile of |f|: -0.83. **By the rule fixed in advance (magnitude at least 0.8) an association is found.**
After the fact, I checked what it means (not pre-registered):
- The statistics are almost monotone in rho: Spearman against rho is +0.986 for the theta1 and theta2 statistics and -1.0 for the theta3 statistic. The signed gap also falls with rho (Spearman -0.83). So the "association" is the same as an association with rho itself.
- Without rho = 26 (the oracle there did not converge, and it is the point with the largest gap) the correlations fall to -0.70 and +0.70, below the threshold.
- The absolute gap against the statistics: -0.46 (theta1) and +0.49 (theta3). Weak.
- The gap is smallest in size at rho = 33 (0.4%), as arXiv 1806.02077 reports for its figure 10, and larger at low rho.

### Literature check on sigma (library only)

A search of all text files in `reference/text` for a sensitivity of an average to sigma found no paper that reports one. The only file that matched (`arxiv-1810.12222`) states that sensitivities were sought with respect to rho. `arxiv-1806.02077` lists sigma among its parameters but reports sensitivities to rho, beta and gamma. The only statement of the sigma discrepancy I know is in the nilss-jax README (not in the library). No web search was made.

### Predictions against results

| Prediction | Result |
|---|---|
| P-B1 controls pass at the first run | **Miss**: C-B1 and C-B2 failed at first, and my angle function had a precision floor (Amendment). |
| P-B2 theta1 and theta2 bounded away from zero | Hit (minimum 0.12 to 0.13). |
| P-B2 minimum of theta1 above 0.3 rad | **Miss**: 0.123 and 0.134. |
| P-B2 theta3 has near-tangency recurrences, P(<0.05) between 0.5% and 10% | Hit (1.1% and 1.05%). |
| P-B3 no criterion explains the gap | Hit (all shifts at most 2e-4, inside the null). |
| P-B4 correlation of log \|f\|_min with \|c\| negative, -0.1 to -0.5 | **Miss**: +0.004. |
| P-B5 gap small (below 0.5%) at rho = 33 | Hit (-0.40%). |
| P-B5 gap above 1.5% at rho = 26 and 28 | Half: 4.47% at 26, **1.34% at 28**. |
| P-B5 no association (magnitude below 0.8) between the gap and the theta3 statistic | **Miss** on the rule: +0.83. Not specific, see above. |
| P-B6 sigma not reported in the library | Hit. |
| P-B7 compute about 10 minutes | **Miss**: about 1.6 minutes. |

### What this shows, and what it does not

- **At rho = 28 the unstable and stable directions stay at least 0.12 rad apart** over about 200000 samples (two orbits). By the Kuptsov-Parlitz criterion that is a splitting that looks bounded away from tangency. The flow direction does approach the unstable direction (theta3 down to 0.007), which is what a trajectory leaving the equilibrium region does.
- **The gap exists at rho = 28 and 30, where those angles are bounded away from zero.** A lack of near-tangencies between the stable and unstable directions does not remove it. At rho = 33 to 40 near-tangencies appear (P(theta1<0.1) up to 0.4%) and the gap does not grow in size.
- **No criterion tested explains the gap** at rho = 28: removing the segments closest to a tangency or to the equilibrium changes the estimate by at most 0.0002 against a gap of 0.0138.
- **The across-rho association is not evidence for the hyperbolicity explanation.** It is a trend with rho.
- **What is left open.** Ni (arXiv 2009.00595) writes the linear response of a hyperbolic system as a shadowing part plus an unstable part, and NILSS computes only the first. That would give a gap even for a perfectly hyperbolic system. This task did not test it. It needs the unstable contribution, which I did not implement. This test also cannot say whether shadowing or finite differences is closer to the true derivative.
- **Limits:** angles sampled every 0.1 time units can miss short dips. The minimum over 2e5 samples is not the infimum. One system, one observable. D3 has 6 points and one oracle that did not converge. CLVs from one algorithm of mine. D2 removes segments as a diagnostic, which is not an estimator.
