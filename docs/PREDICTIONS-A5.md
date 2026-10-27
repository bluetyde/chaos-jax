# Task A5 predictions and gates: our own NILSS, written from the papers

Written and committed before any A5 code ran. Gates follow `TODO.md` (task A5) and are made exact here.
Do not edit a gate or prediction after results. Add a dated Amendment instead.

## Independence

I implement NILSS from `arxiv-1611.00880` (Ni, Wang) and `arxiv-1711.06633` (Schur-complement solve), both read for this task and logged. I have **not** read any nilss-jax source, and I will not read it until this file's results are committed. What I have seen of nilss-jax: README, CHANGELOG, `docs/usage.md`, `docs/reliability.md`. That includes its Lorenz-63 numbers and its settings (dt 0.005, T_seg 0.5, nus 1). My own A2 and A3 runs show nilss-jax gives rho 1.0171, sigma 0.1338, beta -1.6593. My predictions below use them and are **not independent** of them.

## Equation check (done before writing this file)

PDF page images checked: `arxiv-1611.00880` pages 11, 13, 14 (eq. 22 to 24, 34 to 41); `arxiv-1711.06633` page 12 (eq. 48 to 51). By hand: eq. 24 follows from 20, 21, 22 (substituting v*_{i+1} = v*_perp_i - Q b and W_{i+1}(t_{i+1}) = Q); the KKT system (48), the Schur complement (50) and the recovery (51) follow from the Lagrangian (47); the sensitivity formula (41) follows from (15) by Appendix D, and (15) is the LSS formula <dJ.v_perp + eta (J - <J>)> after integrating by parts, with eta = -d xi / dt (C.7). I found **no printed inconsistency**, unlike the LSS paper in Phase 4. One notational point: in eq. 37, R_i maps a_{i-1} to a_i, so R_i is the factor from the QR at the end of segment i - 1 (eq. 30 calls it R_{i}).

## Method (fixed now)

Lorenz-63 (sigma, rho, beta) = (10, 28, 8/3), objective J = z. Parameters rho, sigma, beta differentiated at once, sharing the homogeneous tangents.
For each orbit: start point on the attractor (A2 start points, key `PRNGKey(70)`, then 20 time units of spin-up as in A2), K segments of length T_seg, M homogeneous tangents.
Per segment, integrate the coupled system (u, W, v*) with RK4 at the same dt: du/dt = f, dW/dt = (df/du) W, dv*/dt = (df/du) v* + df/ds. The Jacobian-vector products come from `jax.jvp`. Integrals over a segment (C_i, d_i, the integrals of grad J . W and grad J . v*, and of J) by the trapezoid rule on the steps. At the end of a segment: project onto V_perp, QR of the projected W, b = Q^T v*_perp, then v*_{i+1} = v*_perp - Q b and W_{i+1} = Q (eq. 30, 31, 34, 35). Initial W: random unit vectors in V_perp(u_0), orthonormalised; v*_0 = 0.
Solve the least-squares problem (37) with the Schur complement (50, 51) using a block-tridiagonal solve of my own (`_block_thomas` from `chaos_jax.fixes`). Then xi at segment ends (40) and the sensitivity (41). Lyapunov exponent from the diagonal of R (25).
Settings as in A2 (the nilss-jax documented ones): dt = 0.005, T_seg = 0.5, M = 1, T = 1000.
Main run: the same 32 start points as A2 Stage 2. Expected compute: about 1 to 5 minutes (unknown until the first orbit runs; I time one orbit and state the estimate before the main run).

## Controls first (all before any main sensitivity is read)

- **C-A5.1 Schur solve:** the Schur-complement solution equals a dense solve of the full KKT system (48) on random SPD C and random R, b, d, to relative error 1e-9. Unit test.
- **C-A5.2 tangents:** on a short orbit (T = 1), W matches a finite difference of the primal trajectory in the initial condition, and v* matches a central finite difference in rho, sigma, beta, to relative error 1e-5. Unit test.
- **C-A5.3 continuity:** after solving, the reconstructed v_perp is continuous across every segment interface (eq. 23), to 1e-8 relative to its norm. Unit test.
- **C-A5.4 formula (41) against (15):** for a short orbit, the sensitivity from (41) equals the value from evaluating (15) on the reconstructed v(t) by direct quadrature (eta from xi), to relative error 1e-3 (quadrature). Unit test.
- **C-A5.5 known case:** on the main ensemble, mean leading exponent within 0.02 of 0.9041 and mean z within 0.2 of 23.55.

## Gates (comparison, after the controls pass)

- **A5.1a (strict):** for each of rho, sigma, beta, my ensemble mean agrees with the A2 nilss-jax mean within 2 combined standard errors.
- **A5.1b (practical):** for each, the relative difference to the A2 nilss-jax mean is below 0.5%.
- **A5.2 per orbit:** for the same start points, the median over the 32 orbits of |mine - nilss-jax| / |nilss-jax| is below 1% for rho and sigma, using the per-orbit A2 values.
- **A5.3 against my LSS (A3, T = 320):** relative difference below 1% for each parameter.
- Reported, not gated: the relative difference to the A1 finite-difference oracle.

## Break-it (after committing)

Plant each bug in the working tree, confirm a test fails, restore with git.
1. Drop the xi boundary term in (41).
2. Continuity with the wrong sign: a_{i+1} = R a_i - b.
3. Skip the subtraction Q b when resetting v* (use v*_{i+1} = v*_perp).
Each must be caught by a test among C-A5.1 to C-A5.4 and the comparison tests. A bug nothing catches is reported and the tests are fixed.

## Predictions (not gates)

- **P-A5.1** all unit controls pass at the first run, except that I may have to repair my own code. I count a repair as a miss.
- **P-A5.2 values:** rho 1.017 (range 1.010 to 1.024), sigma 0.134 (range 0.128 to 0.140), beta -1.66 (range -1.68 to -1.64). SEM about 1e-4 for rho and sigma.
- **P-A5.3** A5.1b passes for all three. A5.1a fails for at least two of the three (the standard errors are tiny, as in A3).
- **P-A5.4** A5.2 passes for rho and sigma (median per-orbit difference below 1%).
- **P-A5.5** A5.3 passes for all three.
- **P-A5.6** the break-it bugs are all caught, bug 1 and 3 by the Lorenz comparison and bug 2 by C-A5.3.
- **P-A5.7** the oracle differences are about +1.4% (rho), about -10% (sigma) and about 0% (beta), as for the other two shadowing codes.

## Amendment, 2026-10-27: first run of control C-A5.4

On its first run C-A5.4 **failed**: relative difference 1.016e-3 against the gate 1e-3. The difference fell by half each time I halved dt (1.0e-3, 5.4e-4, 2.8e-4), which is first order, not the second order of the trapezoid error I expected. I traced it with two diagnostics: the per-segment integrals of grad J . (v* + W a) agreed with the direct evaluation to 1e-16, and the quadrature piece sum xi*_i (trapz(dJ/dt) - dJ_i) / T scaled as dt^2 but was only 1.5e-4. The remaining O(dt) part was a bug in **my test**: it started the reconstructed v at zero at t = 0, while the correct start is v*(0) + W(0) a_0 = W(0) a_0 (v* starts at zero but W a_0 does not). The NILSS code was right. I fixed the test and left the gate at 1e-3. The test now gives 1.4e-4 and passes. I count this as one repaired mistake in my own control code (P-A5.1 allowed repairs of my code; I count it as a miss).

## Results (added after the runs, 2026-10-27)

`python -m pytest -q` gives `55 passed, 1 xfailed in 58.81s` (5 new NILSS tests). Commands: `python experiments/phaseA5_run.py probe | run | control | report` (the run takes 4 s for 32 orbits). Output in `results/phaseA5_own.json` and `results/phaseA5_report.json`. I did not read any nilss-jax source, before or after.

### Controls (ran first)

- **C-A5.1** Schur solve against the dense KKT system: pass (relative error below 1e-9).
- **C-A5.2** tangents against finite differences: pass (below 1e-5).
- **C-A5.3** continuity of v_perp at every interface: pass (below 1e-8). Improved after the break-it check, see below.
- **C-A5.4** formula (41) against (15): passes at **1.42e-4** (gate 1e-3) after repairing my test (see the Amendment above).
- **C-A5.5** main run: mean exponent 0.9040 (within 0.02 of 0.9041), mean z 23.555 (within 0.2 of 23.55). Pass.

### Own NILSS, 32 orbits, T = 1000, dt 0.005, T_seg 0.5, M = 1 (same start points as A2)

| | own mean +- SEM (orbit sd) | nilss-jax (A2) | difference | A5.1a (2 combined SE) | A5.1b (below 0.5%) | A5.2 median per orbit | A5.3 against my LSS T = 320 | against the A1 oracle |
|---|---|---|---|---|---|---|---|---|
| rho | 1.01701 +- 0.00006 (0.00035) | 1.01708 +- 0.00007 | -0.007% | pass | pass | 0.03% (pass) | +0.06% (pass) | +1.38% |
| sigma | 0.13363 +- 0.00009 (0.00052) | 0.13375 +- 0.00008 | -0.089% | pass | pass | 0.32% (pass) | -0.50% (pass) | -10.21% |
| beta | -1.65739 +- 0.00112 (0.00631) | -1.65927 +- 0.00116 | +0.113% | pass | pass | 0.31% (not gated) | -0.09% (pass) | +0.05% |

### Break-it (each planted after committing, then restored with git)

| Bug | Tests that failed |
|---|---|
| 1. Drop the xi boundary term in (41) | C-A5.4, Lorenz regression (2 of 5) |
| 2. Wrong sign in the continuity (a_{i+1} = R a_i - b) | C-A5.3, C-A5.4, Lorenz regression (3 of 5) |
| 3. Skip the subtraction Q b at the reset of v* | **First run: C-A5.4 and Lorenz regression, not C-A5.3** (2 of 5). Then I found why: C-A5.3 recomputed the reset from the paper's formula, so it could not see what the code did. The code now records the reset it applied (`V_next`) and C-A5.3 uses it. With bug 3 planted again, C-A5.3, C-A5.4 and the Lorenz regression all fail (3 of 5). |

No bug went uncaught. One control had a blind spot, now fixed.

### Predictions against results

| Prediction | Result |
|---|---|
| P-A5.1 controls pass at first run (repairs count as misses) | **Miss**: C-A5.4 failed at first because of a bug in my test (repaired, see Amendment). |
| P-A5.2 rho 1.017, sigma 0.134, beta -1.66, SEM about 1e-4 | Hit (1.01701, 0.13363, -1.65739; SEM 6e-5, 9e-5, 1.1e-3). |
| P-A5.3 A5.1b passes for all three | Hit. |
| P-A5.3 A5.1a fails for at least two of three | **Miss**: it passes for all three. |
| P-A5.4 A5.2 passes for rho and sigma | Hit (0.03%, 0.32%). |
| P-A5.5 A5.3 passes for all three | Hit. |
| P-A5.6 break-it: bugs 1 and 3 caught by the Lorenz comparison, bug 2 by C-A5.3 | Hit. Bug 3 was not caught by C-A5.3 at first (blind spot, fixed). |
| P-A5.7 oracle differences about +1.4%, -10%, about 0% | Hit (+1.38%, -10.21%, +0.05%). |
| Compute 1 to 5 minutes | **Miss**: 4 s. |

### What this shows, and what it does not

- **A NILSS written from the papers agrees with nilss-jax** at the same settings to 0.007% for rho, 0.09% for sigma and 0.11% for beta in the ensemble mean. Per orbit, for the same start points, the median difference is 0.03% (rho) and 0.3% (sigma, beta), even though the initial tangent vectors differ. That the agreement is closer than I predicted says my predictions underestimated how deterministic NILSS is on Lorenz-63.
- **Three shadowing implementations now give one answer** for rho, sigma and beta: nilss-jax, my own LSS (A3) and my own NILSS. They agree within 0.5%. The LSS and NILSS formulations differ (A3 and A5 solve different problems), and I wrote both from the papers. **All three disagree with the finite-difference oracle** for rho (+1.4%) and sigma (-10%), and agree with it for beta. A3b (noise-regularised finite differences) sided with the oracle.
- **So the sigma gap and the rho gap are not coding errors in any one of these implementations.** They look like a property of shadowing on this system. Candidate cause, not tested: Lorenz-63 at rho = 28 is not uniformly hyperbolic. I did not test it.
- **Independence limits:** I wrote this after reading nilss-jax docs and results (A2, A3, A4), so my code could not be blind to what the answer should be. I did not read its source. All three shadowing codes use the same RK4 and the same per-step quadrature kind. A shared systematic effect of that kind is not excluded.
- **Not done:** other systems (Lorenz-96, Roessler), more than one homogeneous tangent, the FD-NILSS variant, a check against the author's Python reference code (which I did not read).
