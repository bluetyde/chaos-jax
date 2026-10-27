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
