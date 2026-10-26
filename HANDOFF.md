# Handoff log

Append-only. Newest entry last. Format: Status, Changes, Verification, Caveats, Next step.

## 2026-10-26 planning session: repo seeded

- Status: plan, agent rules and an empty scaffold written. No experiment has been run and no phase 1 code exists.
- Changes: `PLAN.md`, `AGENTS.md`, `README.md`, `READING-LOG.md`, `pyproject.toml`, `src/chaos_jax/__init__.py`, `tests/test_env.py`, `reference/` submodule (chaos-reference).
- Verification: see the commit message for the scaffold test result; nothing else was run.
- Caveats: the Lorenz λ₁ value in the plan (about 0.906) comes from memory of the literature and is not confirmed in the reference set; Phase 1 measures it. Whether a cloud session can fetch the `reference/` submodule is untested. The cloud environment's Python version and whether Julia exists are unknown.
- Next step: Phase 1 (core and controls) on branch `claude/phase-1-core`.

## 2026-10-26 Phase 1 session: core and controls

- Status: Phase 1 done. G1.1 to G1.5 all pass. All three break-it bugs were caught. Phase 2 not started.
- Changes:
  - `src/chaos_jax/`: `integrate.py` (RK4 with `jax.lax.scan`), `systems.py` (Lorenz-63, damped oscillator, double pendulum, pendulum energy), `tangent.py` (`jax.jvp` tangent step, `jax.jacfwd` state Jacobian), `lyapunov.py` (Benettin tangent/QR spectrum, and a separate two-trajectory estimator), `cases.py` (fixed choices).
  - `tests/`: `test_systems.py`, `test_integrator.py` (G1.1, G1.2), `test_lyapunov_1_controls.py` (G1.4, run first), `test_lyapunov_2_chaotic.py` (G1.3, G1.5).
  - `experiments/pendulum_dt_scan.py`, `experiments/phase1_measurements.py`, with output in `results/`.
  - `docs/PHASE1-CHOICES.md`: choices written and committed before any Lyapunov run, then the dt scan result.
  - Reference submodule: `git submodule update --init` worked in the cloud. No paper was opened, so `READING-LOG.md` has no new row. Julia was not checked (not needed).
- Verification:
  - `python -m pytest -q` gave: `22 passed in 13.90s` (CPU, limit is 2 minutes).
  - Gate table (G1.1 to G1.5):

    | Gate | Result |
    |---|---|
    | G1.1 order in [3.8, 4.2] | pass on damped oscillator and logistic ODE, every dt halving from 0.1 to 0.0125 |
    | G1.2 AD vs central FD < 1e-6, T=1 | pass for oscillator, Lorenz (state Jacobian and d z / d rho), and pendulum. The oscillator Jacobian also matches `expm` to 1e-6. |
    | G1.3 Lorenz lambda_1 | 0.90640 (start (1,1,1)) and 0.90184 (start (-5,7,30)). Difference 0.0046, limit 0.02. Both in [0.85, 0.95]. Pass. |
    | G1.4 controls | Damped oscillator (c=0.4): exponents -0.2022 and -0.1978, exact value -0.2. Low-energy pendulum: +0.0015 (tangent) and +0.0012 (two-trajectory), limit 0.02. Pass. |
    | G1.5 high-energy pendulum | lambda_1 = 1.3626 (tangent) and 1.3624 (two-trajectory), limit 0.1. Energy drift over T=10 at dt=0.00125 is 2.06e-8, limit 1e-6. Pass. |

  - Lorenz measurement details (T=2000, dt=0.01, from `results/phase1_measurements.json`). The two-trajectory method gives 0.9039 and 0.9043. The full spectrum from start 0 is (0.9064, -0.0009, -14.5721). Its sum is -13.66656. The exact value is -(sigma + 1 + beta) = -13.66667. These values were measured. They were not tuned. The literature figure 0.906 was used only as a band.
  - Pendulum choices: high-energy case theta1 = theta2 = 2.5 rad at rest (E = 23.6), low-energy case theta1 = theta2 = 0.1 rad at rest, g = 9.81, unit masses and lengths. dt = 0.00125. Drift by dt: 0.02 gives 3.3e-3, 0.01 gives 2.3e-4, 0.005 gives 7.2e-6, 0.0025 gives 3.1e-7, 0.00125 gives 2.1e-8. The rule was the largest dt on the ladder with drift at least 10x below the gate. Details in `docs/PHASE1-CHOICES.md`.
  - Break-it results (each run after committing, then restored with `git checkout`):
    1. Flip sign of the beta term in Lorenz: `6 failed, 16 passed`. Caught by `test_g1_3_...`, `test_lorenz_two_trajectory_...`, `test_lorenz_spectrum_sum_...`, `test_lorenz_known_values`, `test_lorenz_fixed_points`, `test_lorenz_divergence_...`.
    2. Wrong dt in one RK4 stage (k3 used dt instead of dt/2): `7 failed, 15 passed`. Caught by both G1.1 tests, the oscillator Jacobian test, the oscillator exponent test, G1.3, the spectrum-sum test and G1.5. G1.2 against finite difference does not catch this bug by itself, because AD and FD both differentiate the same wrong integrator. Only the `expm` comparison caught it there.
    3. Float32 (x64 turned off in `__init__.py`): `15 failed, 7 passed`. With `test_env.py` deselected: `14 failed, 7 passed`. The numerical tests catch it without `test_env`. G1.3 still passes in float32, so it alone would not catch it.
  - No bug went uncaught.
- Caveats:
  - The control test first failed on my own extra check that each damped-oscillator exponent is within 2e-3 of -0.2. Over T=20 the complex pair splits by about 2e-3. I changed that extra check to atol 1e-2 on each exponent and 1e-6 on their sum (exact value -c). The PLAN gate (lambda < 0) was not changed.
  - The pendulum dt is the finest rung of the ladder. The finite-time estimates have not been given error bars beyond the two-start spread.
  - Dependencies: `pip install -e .` was run as root without a virtual environment. Versions: Python 3.13.17, jax 0.12.0.1, numpy 2.5.3, scipy 1.18.1.
- Next step: owner review, then Phase 2 (ground truth for Q_b). Write `docs/PREDICTIONS-2.md`-style gates before any run.

## 2026-10-26 Phase 2 session: ground truth for Q_b

- Status: Phase 2 done. G2.1, G2.2 and G2.3 all pass. Phase 3 not started.
- Changes: `docs/PREDICTIONS-2.md` (committed before any run), `src/chaos_jax/response.py` (time average, ensemble, bootstrap), driven oscillator added to `systems.py`, `tests/test_response_1_control.py`, `tests/test_response_2_lorenz_pipeline.py`, `experiments/phase2_oracle.py`, `results/phase2_oracle.json`, a `READING-LOG.md` row.
- Verification:
  - Control first. `python -m pytest -q -s tests/test_response_1_control.py` gave `3 passed in 3.92s`. G2.3: finite difference 0.37098694, closed form 0.37098692, relative error 5.9e-8 (gate 1e-6).
  - Oracle: `python experiments/phase2_oracle.py` took 1.9 s of compute (N = 1000 start points, 5 rho values, 50 spin-up and 100 averaging time units, dt = 0.01, CPU).
  - **d<z>/drho at rho = 28 = 1.0029, bootstrap SE 0.0021** (central difference over rho = 27 and 29, 2000 resamples). Mean <z>: 22.548, 23.052, 23.551, 24.055, 24.554 at rho = 27, 27.5, 28, 28.5, 29.
  - G2.1: SE is 0.21% of the estimate, gate 10%. Pass.
  - G2.2: quadratic coefficient c2 = -0.0014, SE 0.0032, |c2| < 2 SE. Pass. One-sided slopes: 1.0031 on [27, 28] and 1.0026 on [28, 29].
  - Full suite: `python -m pytest -q` gave `26 passed in 15.95s`.
- Predictions against results:

  | Prediction | Result |
  |---|---|
  | P2.a estimate in [0.95, 1.08] | 1.0029. Hit. |
  | P2.b SE about 0.02 | 0.0021. **Miss by about 10x.** The standard deviation of one 100-unit average of z is 0.095 over start points, not the 0.8 I assumed. |
  | P2.c G2.2 passes | Hit. |
  | P2.d G2.3 error near 1e-8 or smaller | 5.9e-8. **Miss by about 6x.** The gate still passes. |
  | Compute about 3 minutes | 1.9 s. **Miss.** |

- Sanity band: the paper text has no number. Figures 5d and 6a of `arxiv-1204.0159` show about 1.0 to 1.02 (read from a plot, not stated). The oracle value 1.0029 sits at the low end of that band. Nothing was tuned to it.
- Caveats:
  - The oracle uses fixed N, T and h. Because SE was already far below the gate, I did not raise them. A smaller SE would need more start points.
  - Finite difference with h = 1 has an O(h^2) bias. G2.2 found no curvature above the noise. The quadratic term gives a bias of about c2 h^2 = -0.0014, below the SE.
  - The oracle averages 100 time units after a 50-unit spin-up. Phase 3 and 4 horizons T up to 80 are shorter than or close to this window, so the oracle target is the long-time average, as intended.
  - The control window length depends on omega (whole number of periods). This differs from the Lorenz pipeline and is documented in `docs/PREDICTIONS-2.md`.
- Next step: owner review, then Phase 3 (gradient blow-up). Write `docs/PREDICTIONS-3.md` first, including the one number for T* in P3.3.
