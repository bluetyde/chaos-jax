# Task A5c predictions and gates: a map algorithm for the unstable contribution, validated on the Solenoid map

Written and committed before any A5c code ran. Do not edit a gate or prediction after results. Add a dated Amendment instead.

## Scope (owner decision, 2026-10-27)

The earlier plan was to test a missing unstable contribution on Lorenz-63. The owner checked the sources and found that this has no ready algorithm for flows. The smaller scope is:
1. Implement the space-split sensitivity (S3) algorithm for a **map** with a one-dimensional unstable manifold, from arXiv 2103.08816 (Chandramoorthy, Wang).
2. Validate it on the Smale-Williams Solenoid map, which is uniformly hyperbolic with unstable dimension 1 of 3. This is the same ratio as Lorenz-63.
3. Compute the **shadowing contribution** of the same map (the target of shadowing codes) and compare it with the full response. This measures how much a shadowing method misses on a map where theory says the response exists.
4. Decide whether a flow version is worth trying. I do not implement one.

I do not claim anything about Lorenz-63 from this task. A map result cannot show what happens in a non-hyperbolic flow.

## What I read, and what I have seen

Read: arXiv 2103.08816 (sections 3, 4, 6.2 to 6.4, 7.1, 7.2; equations 4.3 to 4.10 checked on the page images), and Ni, arXiv 2003.09801 (the definition of the shadowing direction as the only bounded inhomogeneous tangent solution; lines 225 to 330). Details in `READING-LOG.md`. I wrote the code from the mathematics. I did not look for the authors' source code.
Seen before writing this file: the S3 paper says that for s2 at s = (1, 0) the unstable contribution (in the S3 split, not Ni's split) is almost all of the derivative (the stable part is below 0.01% of it). It prints no numerical values for the Solenoid case, only figures. I use this in P-M3 below, so that prediction is not independent of it.

## The map and the quantities (fixed now)

Solenoid map (Eq. 7.2 of the S3 paper), state (r, theta, z), parameters s = (s1, s2), reference s = (1, 0):
r' = s1 + (r - s1)/4 + cos(theta)/2, theta' = (2 theta + (s2/4) sin(4 theta)) mod 2 pi, z' = z/4 + sin(theta)/2.
Objective J = r squared (the paper's choice, x1 squared plus x2 squared). I work in (r, theta, z) coordinates, where the Jacobian is smooth. Float64 only.
The perturbation field at x_{n+1} = phi(x_n) is chi_{n+1} = (d phi / d s)(x_n) at the reference s. For s1 this is (3/4, 0, 0). For s2 it is (0, sin(4 theta_n)/4, 0).

## The oracle, computed without sampling noise (fixed now)

The theta dynamics does not depend on r, z or s1. It is a one-dimensional expanding circle map. So:
- r_n - s1 = (1/2) sum over k >= 0 of 4^(-k) cos(theta_{n-1-k}), and the expected value of J is s1 squared + 2 s1 E[eps] + E[eps squared] with eps = r - s1.
- **Exact value for s1 at s2 = 0:** theta is uniform, so E[cos theta] = 0 and E[eps] = 0. Then the expected value of J is s1 squared plus a constant, and the derivative with respect to s1 is exactly **2 s1 = 2** at s1 = 1.
- **For any s2:** build the transfer operator of the theta map in a Fourier basis with matrix entries A_kl = (1/2 pi) times the integral over 0 to 2 pi of exp(i (l theta - k T(theta))), by the trapezoid rule on 1024 points (the integrand is periodic and analytic). Take the invariant density as the eigenvector for eigenvalue 1. Autocovariances C(m) = E[cos(theta_0) cos(theta_m)] come from applying A^m to the Fourier coefficients of (density times cos). Sum the geometric series in m and k to 1e-14. Then differentiate the expected value of J with respect to s1 and s2 by central differences with step 1e-4.
- Controls for the oracle (G-M0 below): basis size 48 and 96 agree to 1e-9; the derivative with respect to s1 at s2 = 0 is 2 to 1e-8; the expected value of J agrees with long orbits.

## S3 algorithm (fixed now)

As in the S3 paper, steps 1 to 8, with these choices:
- Many independent orbits, M = 512, each with a run-up of K' = 100 steps (exponentially forgotten) and N = 20 000 averaged steps, plus a tail of 100 extra steps for the lag sum and the backward recursion below. Start points random with seed 5.
- Lags K = 24 terms in the unstable contribution (the paper's reference value in its Figure 9).
- The derivatives (d phi, the second derivatives d2 phi(q, q) and d2 phi(q, v), and d chi q) come from `jax.jvp`. The term d chi_{n+1} q_{n+1} is computed as d(d phi / d s)_n q_n divided by alpha_{n+1}. This follows from chi(x) = (d phi / d s)(phi^{-1} x) and d phi_n q_n = alpha_{n+1} q_{n+1}.
- The scalar c_{n+1} follows from the constraint y_{n+1} . q_{n+1} = - v_{n+1} . p_{n+1}.
- Total response = stable contribution (4.10) + unstable contribution (4.9). The standard error comes from the per-orbit values (SD divided by the square root of M).

## Shadowing contribution (fixed now)

For a map, the shadowing direction is the only bounded solution of v_{n+1} = d phi_n v_n + chi_{n+1} (Ni, Lemma 1). Write v_n = w_n + e_n q_n, where w_n is the regularised tangent solution of (4.6). Then e_{n+1} = alpha_{n+1} e_n + a_{n+1}, so the bounded solution is found by the backward recursion e_n = (e_{n+1} - a_{n+1}) / alpha_{n+1}, started at 0 at the end of the orbit (the error shrinks by 1/alpha per step, and I drop the 100-step tail). The shadowing contribution is the average of dJ_n . (w_n + e_n q_n). The unstable contribution in Ni's sense is the response minus this value.

## Controls first

- **C-M1 derivatives:** on random points, the jvp-based d phi, d2 phi(u, v) and d(d phi / d s) q agree with central finite differences of phi to 1e-6 (relative). Unit test.
- **C-M2 tangent recursions:** q has unit norm and v . q = 0 at every step, to 1e-12. The y recursion keeps y . q = - v . p, to 1e-12. Unit test.
- **C-M3 oracle:** G-M0 below.
- **C-M4 bounded solution:** the shadowing solution v_n = w_n + e_n q_n satisfies the tangent equation exactly (to 1e-12) and stays bounded (norm below 50) on a 2000-step orbit. Unit test.

## Gates (after the controls pass)

All at s = (1, 0) unless stated. SE is the standard error over the 512 orbits.
- **G-M0 (oracle):** (a) basis 48 and 96 agree to 1e-9 for the expected J and its derivatives. (b) The s1 derivative at s = (1, 0) is 2 within 1e-8. (c) The orbit mean of J agrees with the oracle within 3 SE at s = (1, 0) and at s = (1, 0.5). (d) A finite difference of orbit means over s2 = 0.45 and 0.55 (same 512 orbit scheme, no S3) agrees with the oracle derivative at s2 = 0.5 within 3 SE of the difference.
- **G-M1 (stable perturbation, exact value):** the S3 total for s1 is within 3 SE of 2 and within 1% of 2.
- **G-M2 (unstable perturbation):** the S3 total for s2 is within 3 SE of the oracle derivative and within 2% of it.
- **G-M3 (a second point):** the same as G-M2 at s = (1, 0.5) for s2.
- **G-M4 (shadowing, stable perturbation):** the shadowing contribution for s1 is within 1% of 2.
- Reported, not gated: the shadowing contribution for s2 at both points, as a fraction of the oracle value, and the S3 split into its stable and unstable parts.

## Break-it (after committing the code)

Plant each bug, confirm a test or gate fails, restore with git.
1. Drop the d chi term in the y recursion (4.8).
2. Flip the sign of the unstable contribution (4.9).
3. Drop the a p term in (4.8).
4. In the shadowing recursion, use e_n = (e_{n+1} + a_{n+1}) / alpha_{n+1} (wrong sign).
Each bug must make a unit test or the small-sample gate test fail. A bug that nothing catches is reported and the tests are fixed.

## Predictions (not gates)

- **P-M1** all controls pass at the first run, except that I may have to repair my own code (a repair counts as a miss).
- **P-M2** G-M0, G-M1, G-M2 and G-M3 pass. S3 total for s1 is 2.000 within 0.01. For s2, the S3 total agrees with the oracle within 1%. I give this about 70%, because two recursions (p and y) are easy to get wrong, and the paper's equations had no printed error that I found.
- **P-M3** the shadowing contribution for s2 differs from the full response by more than 5% of it at s = (1, 0), about 60%. Reason: the S3 paper says the unstable contribution dominates for s2, but that is its own split, and the split of Ni is different. I do not know which way the difference goes or how large it is.
- **P-M4** the shadowing contribution for s1 equals the full response (the perturbation is stable), within 1%.
- **P-M5** the run time of the main S3 pass (512 orbits, 20 200 steps, two parameters, two points) is 2 to 10 minutes on 4 CPU cores. A first probe gives the real number.

## What the result will and will not show

- If the shadowing contribution misses a large part of the response for s2 on the Solenoid map, then on a uniformly hyperbolic map with unstable dimension 1 of 3 a shadowing code has a systematic error for some perturbations. That shows the mechanism is real. It does not show that the +1.4% (rho) and -10% (sigma) gaps on Lorenz-63 come from it. The Lorenz-63 flow is not uniformly hyperbolic (A5b found no sign of near-tangency at rho = 28, but that is not a proof).
- If the shadowing contribution matches the response for both parameters, that says unstable dimension 1 of 3 alone does not produce a gap for these perturbations and this objective.
- Ni states two conditions for a small unstable contribution (fast decay of correlations, and an objective gradient and perturbation not aligned with the unstable direction). The test is one objective and two perturbations. It does not test those conditions in general.

## Decision rule for a flow version (written now)

I will recommend trying a flow version only if all three hold: (a) G-M2 and G-M3 pass, (b) the shadowing contribution differs from the full response by more than 1% for s2 (so the unstable part matters), and (c) a source in the library gives a flow algorithm that does not need hyperbolicity. The owner reports that none does: Ni and Tong call their flow formula a building block, and say that Lorenz systems are not hyperbolic and that the algorithm can explode. So I expect to recommend against a flow version in this task, and I list what a flow version would need.

## Amendment, 2026-10-27 (before any S3 run): derivative stencil of the oracle

While building the oracle I checked gate G-M0 (a). With the two-point central difference of step 1e-4 that I wrote in the oracle section, the s2 derivative at s = (1, 0.5) differs by 1.2e-7 between basis 48 and basis 96. That **fails** the 1e-9 in G-M0 (a). The cause is roundoff of about 1e-11 in the expected value, divided by the small step. The expected value itself agrees between the two bases to 2e-12.
Change (a choice of method, made before any S3 result exists): the s2 derivative of the oracle uses the four-point central stencil (-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)) / (12 h) with h = 0.02. The s1 derivative uses the exact formula 2 s1 + (4/3) E[cos theta]. The gate stays at 1e-9 for the expected value. For the derivatives, the check with the new stencil gives a difference between the two bases of 2.2e-10 at s2 = 0 and 5.7e-10 at s2 = 0.5, so the 1e-9 holds and I do not change it. The exact value at s2 = 0 is -23/60 = -0.38333...; the stencil gives -0.3833333 to about 1e-8 (truncation of the stencil). The oracle precision only needs to be far below the standard error of S3 (about 1e-3).
