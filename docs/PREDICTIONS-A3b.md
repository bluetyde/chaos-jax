# Task A3b predictions and gates: a third method, noise-regularised finite differences

Written and committed before any A3b code ran. This is a follow-up to A3 that the owner asked for. It adds no gate to `TODO.md` and changes no earlier result.
Do not edit a gate or prediction after results. Add a dated Amendment instead.

## Question

A3 left this open. nilss-jax and my own LSS agree with each other: rho about 1.017, sigma about 0.134. The finite-difference oracle (A1, A1b) gives rho 1.003 and sigma 0.149, and it is stable against h, window, dt and spin-up. Which side is closer to the true derivative of the long-time average for the deterministic Lorenz-63?

## Method and why I chose it

A third method must not be a finite difference of the deterministic ensemble average, and must not be shadowing.
I considered the ensemble tangent (Ruelle formula) of Lea et al. 2000 and Eyink et al. 2004. Eyink et al. report heavy tails and about 2.5e8 samples for 10% accuracy at averaging time 2 (as read in the log). I need about 1%, so that is out of reach for my compute.
Instead: **noise-regularised finite differences.** Add small additive white noise to Lorenz:
dX = F(X; theta) dt + eps dW.
For eps > 0 the invariant measure is smooth, and the response of the average to the parameter is a smooth function of the parameter. A central difference at a moderate h then measures the derivative without roughness bias. The standard assumption (stochastic stability) is that this derivative tends to the deterministic one as eps goes to 0. I do not prove that assumption. I look at how the derivative changes as eps falls.
This is a different kind of evidence from A1 and A3, and it rests on its own assumption.

Scheme: one step is an RK4 step for the drift, then add eps sqrt(dt) xi with xi a standard normal vector in three dimensions. This is weakly first order in the noise. I use the same scheme at theta - h and theta + h, so its error largely cancels in the difference. dt = 0.01. I do not claim it has no bias.

## Setup (fixed now)

- Parameters and steps: rho with h = 1 and h = 0.5; sigma with h = 1 and h = 0.5.
- Noise levels eps = 4, 2, 1, 0.5, 0.25, 0.125, and eps = 0 as the anchor. (The state is of size 10 to 30, and the drift is about 50 per unit time. eps = 4 is large and eps = 0.125 is small.)
- Start points: x, y uniform in [-15, 15], z uniform in [5, 45]. 50 time units of spin-up, then a time average of z over 100 time units. N = 2000 start points for h = 1 and 8000 for h = 0.5 (as the A1 rungs).
- Noise: independent per start point, and the same noise sequence at theta - h and theta + h (paired). Key: `fold_in(PRNGKey(95), 100 x parameter_index + 10 x noise_index + h_index)`.
- Slope: (mean z(+h) - mean z(-h)) / (2 h). Bootstrap SE: 2000 resamples over start points.
- Expected compute: about 10 minutes in all (up to 25). I run one job at a time. I note the speed from the control and re-check this estimate before the main run.

## Controls first

- **C-N1, noisy integrator:** an Ornstein-Uhlenbeck process dx = -k x dt + eps dW with k = 1, eps = 1, scheme as above (RK4 drift, then noise), dt = 0.01. The mean square of x has the exact stationary value eps^2 / (2 k), so d<x^2>/dk = -eps^2 / (2 k^2) = -0.5. The pipeline (random start points, spin-up, window average, central difference in k with h = 0.1, N = 4000, window 100, spin-up 50) must give a slope within 3% of -0.5. The scheme has an O(dt) bias of about 0.5%, so 3% is a margin for noise plus bias.
- **C-N2, eps = 0 anchor:** the eps = 0 runs reproduce the A1 slopes at the same h within 2 combined SE: rho 1.00331 +- 0.00151 (h = 1) and 1.00305 +- 0.00150 (h = 0.5); sigma 0.15145 +- 0.00138 (h = 1) and 0.14887 +- 0.00142 (h = 0.5).
- If either fails I stop and find out why. I read no noisy-Lorenz result before they pass.

## Gates

- **N1, h-convergence at each eps:** |slope(h = 1) - slope(h = 0.5)| <= 2 sqrt(SE^2 + SE^2). I report which eps pass. If the pair disagrees at an eps, I take no conclusion from that eps.
- **N2, decision for each of rho and sigma.** Let S be the slope at eps = 0.125 and h = 0.5, with SE s. This is the smallest eps; it is only used if N1 passes at eps = 0.125 and eps = 0.25.
  - Finite-difference side: |S - F| <= 2 sqrt(s^2 + se_F^2) and |S - Sh| > 2 sqrt(s^2 + se_Sh^2).
  - Shadowing side: the reverse.
  - Otherwise: unresolved.
  F = 1.00317 +- 0.00074 for rho and 0.14882 +- 0.00072 for sigma (A1 weighted means). Sh = 1.01708 +- 0.00007 for rho and 0.13375 +- 0.00008 for sigma (nilss-jax, A2).
- **N3, trend:** I report the slope against eps for rho and sigma, with SE. I say whether the slopes at eps = 0.125, 0.25 and 0.5 are consistent with each other (within 2 SE), since only a plateau supports an eps-to-0 reading.

## Disclosure

I have seen the nilss-jax numbers (see `docs/PREDICTIONS-A2.md`) and my own A1, A1b, A3 results. My predictions below are guesses made with them in mind.

## Predictions (not gates)

- **P-N1 controls:** C-N1 within 3% (I expect about 0.5%), C-N2 passes.
- **P-N2 trend:** the slope of rho changes slowly with eps. At eps = 4 it is lower or higher by up to 0.03 than at eps = 0.125, and it is flat within 2 SE for eps <= 0.5. Same for sigma, with changes up to 0.01.
- **P-N3 decision, rho:** **finite-difference side**: S = 1.004 +- 0.004 at eps = 0.125. I give this about 65% (roughly two in three) against 25% for the shadowing side and 10% for "unresolved". The reason is that A1 and A1b found no trend or window effect in the deterministic differences, and a smooth response at small eps should look like them.
- **P-N4 decision, sigma:** finite-difference side as well, S = 0.148 +- 0.004, with the same odds.
- **P-N5 N1:** h-convergence holds at eps >= 0.5 and may fail at small eps if the response is rough there. I give about 60% that it holds at every eps.
- **P-N6 compute:** about 10 minutes.
