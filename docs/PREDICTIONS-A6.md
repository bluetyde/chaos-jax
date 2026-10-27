# Task A6 predictions and gates: does a long-time average of the double pendulum depend on the start point?

Written and committed before any A6 run. Do not edit a gate or prediction after results. Add a dated Amendment instead.
`TODO.md` (task A6) says: check first that the long-time average does not depend on the starting point. A Hamiltonian flow with a mixed phase space may fail this. If it fails, say so and stop this task. This file covers that check only (stage 1). Stage 2 (sensitivity) starts only if stage 1 passes, and gets its own predictions first.

## Why the check needs an energy shell

The double pendulum conserves energy. There is no attractor. A long-time average can depend on the energy, so "independent of the start" can only hold for starts with the **same energy**. I fix the energy E = 23.58 (the Phase 1 high-energy case, theta1 = theta2 = 2.5 rad at rest, g = 9.81, unit masses and lengths, lambda1 = 1.36 measured there). On this energy shell the motion may still split into a chaotic sea and regular islands. Starts inside an island give a different average. That is what the check looks for.

## Method (fixed now)

- System: `chaos_jax.systems.double_pendulum`, parameters (g, l1, l2, m1, m2) = (9.81, 1, 1, 1, 1). RK4 at dt = 0.00125 (the Phase 1 choice, energy drift 2e-8 over T = 10). Float64.
- Starts: 32 points on the energy shell E = 23.58. Draw theta1, theta2 uniform on (-pi, pi). Keep a draw only if the potential energy is below E. Draw a random direction for (w1, w2) and scale it so the total energy equals E exactly. Seed `numpy.random.default_rng(60)`. No spin-up: there is no attractor to reach, and the starts are already on the shell.
- Observable (chosen now): J = cos(theta2). It is bounded in [-1, 1]. The long-time average is the mean over T = 1000 time units. Reported but not gated: cos(theta1).
- Block means: the mean of J over blocks of 50 time units (20 blocks per start). Within-start standard error SE_i = SD of the 20 block means divided by sqrt(20). The first 5 blocks give the T = 250 average.
- Finite-time Lyapunov exponent per start (reported, not gated): `lyapunov_spectrum`, T = 200, 100 blocks discarded.
- Expected run time: the main scan is 32 orbits of 800 000 steps. I expect 1 to 5 minutes on 4 CPU cores, and the Lyapunov step about 1 minute. Controls take about 10 seconds.

## Controls first

- **C-A6.1 shell:** every start has relative energy error below 1e-12.
- **C-A6.2 integrator:** for 4 starts at T = 100, relative energy drift below 1e-6 at the end. If it fails, the spread test is not trusted and I stop.
- **C-A6.3 drift over the full run:** for all 32 starts at T = 1000, final relative energy drift below 1e-4. If it fails, I say that the spread test is suspect and report the largest drift. I do not change dt and rerun without a dated Amendment.

## Gates

- **G-A6.1 (size of the spread):** R = (SD of the 32 start means at T = 1000) / (mean of the 32 within-start SE at T = 1000) is below 2.0.
- **G-A6.2 (the spread falls with time):** SD_between at T = 1000 divided by SD_between at T = 250 is below 0.7. For independent noise I expect about 0.5. A ratio near 1 means a real difference between starts.
- **G-A6.3 (no outlier):** no start has a mean farther than 4 times its own SE from the median of the 32 means.

Stage 1 passes only if G-A6.1, G-A6.2 and G-A6.3 all pass. If any gate fails, I report "not start-independent on this energy shell". I list the starts that differ, with their finite-time exponents, and I stop A6.

## Predictions (not gates)

- **P-A6.1** stage 1 passes (all three gates), about 60%. Reason: at E = 23.58 the energy is high, and most of the shell should be chaotic. Small regular islands may remain. With 32 random starts, an island that holds 3% of the shell volume is missed with probability 0.97^32 = 38%, so a pass does not prove there are no islands.
- **P-A6.2** the finite-time exponent is above 0.5 for at least 30 of the 32 starts.
- **P-A6.3** the 32 means of cos(theta2) at T = 1000 lie within 0.05 of each other (range), and the ensemble mean is within 0.3 of zero. This is a guess, since I have not computed it.
- **P-A6.4** C-A6.3 passes for all starts (drift below 1e-4).

## Known problem for stage 2 (written now, so it is not a surprise later)

If the parameter changes, the energy of a fixed start changes. A sensitivity then needs a definition: the average at fixed energy E, with starts rebuilt on the shell for each parameter value, or the average at fixed start. Shadowing theory assumes a hyperbolic attractor. A Hamiltonian flow has neither. So stage 2 may have no meaningful answer for LSS and NILSS even if stage 1 passes. If stage 1 passes, I propose the definition and the risk to the owner before running anything.

## Results (stage 1; written after the run, gates unchanged)

Files: `results/phaseA6_controls.json`, `phaseA6_spread.json`, `phaseA6_report.json`. Script: `experiments/phaseA6_spread.py`. Run time: scan 9.9 s, Lyapunov step 9.7 s (faster than the 1 to 5 minutes I stated).

**Controls.** C-A6.1: energy error 4.5e-16, pass. C-A6.2: drift 5.3e-8 at T = 100, pass. C-A6.3: largest final drift 2.2e-6 over T = 1000, pass.

**Gates, observable cos(theta2), 32 starts, T = 1000.**

| Gate | Value | Limit | Result |
|---|---|---|---|
| G-A6.1 R = SD between starts / mean within-start SE | 1.52 (0.0195 / 0.0129) | below 2.0 | pass |
| G-A6.2 SD(T = 1000) / SD(T = 250) | 0.67 | below 0.7 | pass (barely) |
| G-A6.3 largest distance from the median in own SE | 138 | below 4 | **fail** |

**Stage 1 fails.** The failing start is start 22: (theta1, theta2, w1, w2) = (-1.240, 0.209, -2.151, 8.620). Its finite-time exponent is 0.015, against 1.19 to 1.56 for the other 31. The block means of cos(theta2) on that start hardly change (SE 0.0006), and its mean is -0.023, against a median near 0.06. For the second observable, cos(theta1), the difference is large: start 22 gives 0.805 and the others give about 0.05, so G-A6.1 (R = 10.6) and G-A6.2 (ratio 0.99) fail for that observable too. This is the signature of a regular island: one orbit that does not mix with the chaotic sea.

**Post-hoc check (not a gate, chosen after I saw the failure).** Without start 22 (the 31 starts with exponent above 0.5), cos(theta2) gives R = 0.93, a ratio of 0.48 (the value I expect for independent noise), a largest distance of 2.2 SE, and a range of 0.052. For cos(theta1): R = 1.07, ratio 0.48, largest distance 3.3 SE. So the chaotic sea looks start-independent at this resolution. I did not predict this, and a rule that drops starts by their exponent was not in the plan.

**Decision.** Following `TODO.md`, I say that the long-time average on this energy shell is **not start-independent** and I stop A6. Stage 2 (sensitivity to a parameter) is not started.

**Predictions against results.**

| Prediction | Outcome |
|---|---|
| P-A6.1 stage 1 passes (about 60%) | **Miss.** G-A6.3 failed. |
| P-A6.2 exponent above 0.5 for at least 30 of 32 starts | **Hit.** 31 of 32. |
| P-A6.3 range of the 32 means within 0.05 and ensemble mean within 0.3 of zero | **Partly.** The mean is 0.061 (hit). The range is 0.108 with start 22 (miss) and 0.052 without it. |
| P-A6.4 drift below 1e-4 for all starts | **Hit.** Largest 2.2e-6. |

**What this means.**
- At E = 23.58 the phase space is mixed. One of 32 random starts (about 3%) lay in a regular island. With 32 starts, the share of the shell that is regular is only known roughly: one hit in 32 is consistent with anything from about 0.1% to 16%. This is my estimate from the count, not a measurement of the island size.
- A sensitivity ensemble that draws random starts would give each start a 3% chance (about) of an island average. Shadowing theory does not apply on an island at all.
- A dependable stage 2 would need a rule to exclude regular starts (for example by their exponent), a definition of the average at fixed energy when a parameter changes, and a reason to trust shadowing codes on a flow that is not hyperbolic. I did not attempt any of these. They are the owner's choice.
- Limits: one energy, one observable pair, 32 starts, one dt, one seed. Other energies may behave differently. I did not scan the energy.
