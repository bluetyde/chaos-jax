# Phase 1 choices

Written before any Lyapunov exponent was measured. The gates G1.1 to G1.5 are in `PLAN.md`
and are not changed here.

## Fixed before measurement

| Item | Choice | Reason |
|---|---|---|
| Lorenz dt | 0.01 | Same value as the Phase 4 note in `PLAN.md`. |
| Lorenz start points | (1, 1, 1) and (-5, 7, 30) | Far apart. Each runs 50 time units first to reach the attractor. |
| Lorenz averaging time | 2000 time units | Long enough to make the finite-time spread small. |
| Pendulum high-energy case | theta1 = theta2 = 2.5 rad, both at rest, g = 9.81, unit masses and lengths | Large amplitude, so the second arm can flip. Energy E = -3 g cos(2.5) = 23.6. |
| Pendulum low-energy case | theta1 = theta2 = 0.1 rad, both at rest | Small amplitude, so motion is close to two normal modes. |
| Pendulum averaging time | 1000 time units | A non-chaotic system gives log-growth about ln(t) / t, so a long time is needed for it to fall below 0.02. |
| Benettin renormalisation | every 10 steps, first 100 blocks left out of the average | Standard. |
| Damped oscillator | c = 0.4, k = 4 | Underdamped. Exact exponent is -c / 2 = -0.2. |

## Rule for the pendulum dt

Try dt in the ladder 0.02, 0.01, 0.005, 0.0025, 0.00125. For each, integrate the high-energy
case to T = 10 and compute max |E(t) - E(0)| / |E(0)|. Choose the largest dt whose drift is at
least 10 times below the G1.5 gate of 1e-6. The 10x margin is my choice. It stops the test
from passing by a hair.
