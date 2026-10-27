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

## Results (added after the runs, 2026-10-27)

Commands: `python experiments/phaseA3b_noise.py run rho` (13.3 min), `run sigma` (13.5 min), `control`, `report`. Summary in `results/phaseA3b_noise.json`. The raw arrays are not committed. They are regenerated by `run`.
`python -m pytest -q` gives `47 passed, 1 xfailed in 51.37s`. Three new tests: the noisy integrator at eps = 0 equals the deterministic average, the key gives repeatable noise, and the OU control.

### Controls (ran first)

- **C-N1** OU slope in k: -0.5080 against the exact -0.5. Relative error 1.6% (gate 3%). Pass.
- **C-N2** eps = 0 anchor against A1: rho h = 1: +1.00413 +- 0.00147 against +1.00331 (0.39 SE); h = 0.5: +1.00125 +- 0.00145 against +1.00305 (0.86 SE); sigma h = 1: +0.14927 +- 0.00138 against +0.15145 (1.12 SE); h = 0.5: +0.14863 +- 0.00143 against +0.14887 (0.12 SE). All pass.

### Slopes against the noise level (h = 0.5 unless stated; bootstrap SE about 0.0015 for eps <= 0.25, 0.003 to 0.004 at eps >= 1)

| eps | rho h = 1 | rho h = 0.5 | sigma h = 1 | sigma h = 0.5 |
|---|---|---|---|---|
| 4 | 0.9235 +- 0.0036 | 0.9245 +- 0.0036 | 0.1007 +- 0.0036 | 0.1057 +- 0.0036 |
| 2 | 0.9243 +- 0.0040 | 0.9272 +- 0.0041 | 0.1232 +- 0.0042 | 0.1261 +- 0.0042 |
| 1 | 0.9674 +- 0.0035 | 0.9678 +- 0.0031 | 0.1467 +- 0.0033 | 0.1456 +- 0.0032 |
| 0.5 | 0.9947 +- 0.0021 | 0.9923 +- 0.0020 | 0.1521 +- 0.0019 | 0.1486 +- 0.0020 |
| 0.25 | 1.0012 +- 0.0016 | 1.0003 +- 0.0016 | 0.1517 +- 0.0015 | 0.1490 +- 0.0016 |
| 0.125 | 1.0003 +- 0.0016 | **1.0012 +- 0.0015** | 0.1493 +- 0.0015 | **0.1466 +- 0.0014** |
| 0 | 1.0041 +- 0.0015 | 1.0013 +- 0.0015 | 0.1493 +- 0.0014 | 0.1486 +- 0.0014 |

### Gates

- **N1 (h = 1 and h = 0.5 agree within 2 combined SE): pass at all 7 noise levels for both parameters** (14 of 14).
- **N2 (decision at eps = 0.125, h = 0.5):** rho **finite-difference side** (S = 1.0012 +- 0.0015: within 2 SE of 1.00317, and 10 SE from 1.01708). sigma **finite-difference side** (S = 0.1466 +- 0.0014: within 2 SE of 0.14882, and 9 SE from 0.13375).
- **N3 (plateau for eps = 0.5, 0.25, 0.125):** sigma **yes**. rho **no**: the slope at eps = 0.5 (0.9923) is 3.6 SE below the one at eps = 0.125 (1.0012). The rho slope does plateau for eps <= 0.25 (1.0003, 1.0012 and 1.0013 at eps = 0).

### Predictions against results

| Prediction | Result |
|---|---|
| P-N1 controls pass (OU about 0.5%) | Hit on pass. The OU error is 1.6%, not 0.5%. |
| P-N2 slope at eps = 4 within 0.03 of eps = 0.125, flat for eps <= 0.5 (sigma: up to 0.01) | **Miss.** rho at eps = 4 is 0.077 lower and sigma is 0.041 lower. rho at eps = 0.5 is 3.6 SE below the plateau. |
| P-N3 rho: finite-difference side, S = 1.004 +- 0.004 | Hit (1.0012). |
| P-N4 sigma: finite-difference side, S = 0.148 +- 0.004 | Hit (0.1466). |
| P-N5 N1 holds at every eps (about 60%) | Hit. |
| P-N6 compute about 10 minutes | **Miss.** 27 minutes, above the stated upper bound of 25. |

### What this shows, and what it does not

- **Noise reduces the sensitivity, and the reduction vanishes as the noise falls.** rho: 0.925 at eps = 4, 0.968 at eps = 1, 1.001 at eps <= 0.25. sigma: 0.106 at eps = 4, 0.146 at eps = 1, about 0.149 at eps <= 0.5. At every eps the two steps agree (N1), so the smooth-response reading is supported by the data.
- **The limit as eps goes to 0 is the A1 value, not the shadowing value.** The smallest-eps slopes are within 2 SE of the A1 oracle for both parameters, and 9 to 10 SE away from nilss-jax. Nothing in the trend moves toward 1.017 or 0.134.
- **So under the stochastic-stability assumption, the finite-difference derivative at eps = 0 is the better estimate, and the two shadowing codes (nilss-jax and my LSS) are about 1.4% high for rho and about 10% low for sigma.** The assumption is that the derivative at small noise is continuous at eps = 0. I did not test that. The data show a plateau for eps <= 0.25, and that is all.
- **This is still a finite difference of averages.** The noise makes the response smooth for eps > 0, so a finite difference there is a derivative by construction. At eps = 0 it adds nothing beyond A1. The new evidence is the trend, not the eps = 0 point.
- **I did not find why the shadowing codes differ.** Lorenz-63 at rho = 28 is not uniformly hyperbolic (the shadowing derivation assumes it). That is the usual suspect and I did not test it.
- **Limits:** one noise type (additive, isotropic), one dt (0.01) and a first-order noise scheme, one parameter point, rho and sigma only, smallest eps 0.125. The A1b window, dt and spin-up checks cover the eps = 0 anchor but not the noisy runs.
