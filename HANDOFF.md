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

## 2026-10-26 Phase 2 follow-up: 25 to 50 slope check

- Status: done on branch `claude/phase-2-wide-slope` (stacked on `claude/reading-notes`, which is stacked on the Phase 2 branch). No Phase 3 work.
- Changes: `docs/PREDICTIONS-2b.md` (predictions and decision rule committed before the run, results appended after), `experiments/phase2b_wide_slope.py`, `results/phase2b_wide_slope.json`.
- Verification: `python experiments/phase2b_wide_slope.py` took 39 s. Ensemble line slope over rho 25 to 50 is 0.9888 (SE 0.0001). A single run with the Lea et al. setup gives 0.9925. Across 20 start points the slope spans 0.9698 to 0.9925. `python -m pytest -q` gave `26 passed in 16.26s` (no source change in this follow-up).
- Result: the range explains part of the gap to 0.96 (decision rule band 0.98 to 1.00). About 0.03 remains unexplained. Window slopes vary from 0.963 to 1.007. All three predictions hit.
- Caveats: the Lea et al. fit range is only "roughly 25 to 50" in the text, and I did not test wider ranges. The SE of 0.0001 on the line slope is sampling error over start points only. It does not include the bias from the fixed grid or from RK4. No Phase 2 number changed.
- Next step: owner decides whether the remaining 0.03 matters. Phase 3 is still next.

## 2026-10-26 Phase 3 session: gradient blow-up

- Status: Phase 3 done. P3.1, P3.2 and P3.3 pass. **P3.4 fails** as the plan states it. Phase 4 not started.
- Changes: `docs/PREDICTIONS-3.md` (committed before any run, results appended), `src/chaos_jax/blowup.py`, `tests/test_blowup.py`, `experiments/phase3_blowup.py`, `results/phase3*.json` and `.npz`, a Phase 9 proposal in `PLAN.md` (owner request, no gates). This branch also carries the reading notes and the 25 to 50 slope check, which were not yet merged.
- Verification:
  - `python -m pytest -q` gave `30 passed in 19.75s`.
  - Controls first: oscillator slope -0.2004. Low-energy pendulum slope -0.0146 (limit 0.02).
  - Lorenz Jacobian log-norm slope over T in [2, 20]: **0.9057** (SE 0.0037, N = 200). Phase 1 lambda_1 mean 0.9041. Difference +0.18%.
  - Naive gradient of the time average of z, median over 200 start points: 0.81 at T = 1, 13.5 at T = 8 (so **T\* = 8**), 105 at T = 10, 4.6e5 at T = 20, 2.0e9 at T = 30. Oracle 1.0029.
  - Sign of the gradient: negative for 37.5% of start points at T = 5 and 53% at T = 30.
  - Float32 against float64: median relative difference 1.6e-2 at T = 10, 0.21 at T = 13, so **T_leave32 = 14**. Float64 against the oracle is already 40% off at T = 1, so **T_leave64 = 2**. P3.4 therefore fails.
- Predictions against results: see the table in `docs/PREDICTIONS-3.md`. Misses: pendulum slope sign, T* (8 against 6), T_leave64 (2 against 4), T_leave32 (14 against 16). All numbers except the pendulum sign were inside my stated ranges. I predicted P3.4 would fail.
- Caveats:
  - P3.4 compares two times defined by my own rule (fixed before the run). The float64 side is the bottom of the grid because of short-average bias. It does not show a clean cross-over.
  - One system (Lorenz). The pendulum is Phase 6.
  - The float32 run uses the same code with x64 turned off in a separate process.
  - The median |g| is a size. It is not an error, because the signs are random.
- Next step: owner review. Then Phase 4 (the fixes). It needs a sparse or block-tridiagonal solver for LSS, and the formulation papers `arxiv-1204.0159` and `arxiv-1611.00880` (check equations against the PDF pages). Write `docs/PREDICTIONS-4.md` first.

## 2026-10-26 Phase 4 session: the fixes

- Status: Phase 4 done, with one failed gate reported. **LSS fails the driven-oscillator control** (error 4.5e-2 against the gate 1e-4). All other methods pass the control. On Lorenz, LSS works at all five horizons, truncated and ensemble-of-truncated work at most of them, and naive and ensemble-of-naive work at none. NILSS (stretch goal) was not attempted. Phase 5 not started.
- Changes: `docs/PREDICTIONS-4.md` (predictions, gates and equation check committed before any code ran, results appended), `src/chaos_jax/fixes.py` (naive, truncated, block-tridiagonal LSS), `src/chaos_jax/controls.py`, `driven_oscillator_phase` in `systems.py`, `tests/test_fixes_*.py`, `experiments/phase4_fixes.py`, `results/phase4*`, `READING-LOG.md` rows.
- Verification:
  - `python -m pytest -q` gave: `40 passed, 1 xfailed in 41.82s`. The xfail is the LSS control gate test (strict), kept on purpose.
  - Control first: naive error 1.9e-11, truncated with whole window identical to naive, LSS 4.49e-2 (fail). LSS error falls like 1/T: 4.5e-2, 1.1e-2, 2.5e-3 at 10, 40, 160 periods. LSS also matches a dense solve of the full constrained problem.
  - Lorenz, oracle 1.0029, median (IQR / oracle) at T = 80: naive 1.1e25 (7e28), truncated 0.853 (0.83), ensemble naive 3.8e28, ensemble truncated 0.851 (0.60), LSS 1.016 (0.003). "Works" per gate: LSS at T = 5, 10, 20, 40, 80. Truncated at T = 5, 20, 40, 80 (not 10, IQR 1.08). Ensemble truncated at all five. Naive and ensemble naive at none.
  - Cost at T = 80, one estimate: naive 2.8 ms, truncated 2.9 ms, LSS 79 ms. Memory above a bare process: 50 MB, 51 MB, 105 MB.
  - Break-it: all three caught (3, 1 and 3 failing tests). Bug 2 (shifted window) was caught only by the Lorenz test at first. I added a test that now catches it directly.
- Predictions against results: see the table in `docs/PREDICTIONS-4.md`. Misses: LSS fails the control, LSS works at T = 5, LSS IQR 25 times smaller than predicted, truncated passes at more horizons than predicted, ensemble naive fails at T = 5, cost and compute were much lower than predicted.
- Equation check: the printed LSS equations in arXiv 1204.0159 (Eq. 16 to 21) disagree with each other in four places. I derived my own discretization from Eq. 13 and 15. Details in `docs/PREDICTIONS-4.md`.
- Caveats:
  - **LSS and the Phase 2 oracle differ by 1.3%** (1.016 against 1.0029). The gate (25%) does not see it. The cause is not known. A central difference with h = 1 being a coarser slope than the local derivative is one candidate. Not tested.
  - The LSS control gate cannot be met on a window of 10 periods. I did not change it.
  - alpha = 10 was fixed on dimensional grounds. A sweep shows little effect (0.2% for alpha 3 to 30).
  - Truncation length L = 2.0 was fixed in advance. Other L values were not run.
  - The truncated gate verdict at T = 10 is near the threshold and flips on noise.
  - Ensembles reuse the same 5000 start points for every T.
  - The control for (b) is a code check only.
- Next step: owner review. Then Phase 5 (fitting task). Decide first whether to run the oracle at smaller h to explain the 1.3% gap (needs a prediction file first), and whether to attempt NILSS. Read `arxiv-1611.00880` before NILSS.

## 2026-10-26 planning session: PR 4 merged, TODO.md written

- Status: PR 4 (Phase 4) merged at the owner's request after a quick check (tests: 40 passed, 1 xfailed in 20.84s on the PR branch; predictions committed 22:39:53, before the first code at 22:44:12). Direction changed to an independent audit of nilss-jax (owner approved). `TODO.md` holds the new task list (A0 to A8).
- Changes: `TODO.md` (new), `AGENTS.md` (read-list now includes `TODO.md`), `PLAN.md` (prior-art update paragraph). No code changed.
- Verification: planning session re-ran the PR 4 tests only; it did not re-check the Phase 4 numbers (LSS 1.016, oracle 1.0029).
- Caveats: the nilss-jax claims in `TODO.md` come from its README, changelog and doc headings, unverified; Phase 9 text in `PLAN.md` was added by a cloud session at the owner's request and has no search behind it yet.
- Next step: Phase A0 (tidy) then A1 (oracle v2) on branch `claude/phase-A1-oracle-v2`.

## 2026-10-26 Task A1 session: oracle v2

- Status: A1 done. Control passes. rho and sigma converged. beta not converged under the gate (one pair fails). A2 not started.
- Changes: `docs/PREDICTIONS-A1.md` (predictions and gates committed before any code, results appended), `tests/test_oracle_v2_control.py`, `experiments/phaseA1_oracle_v2.py`, `results/phaseA1_oracle_v2.json`, `.gitignore` (raw arrays not committed).
- Verification:
  - `python -m pytest -q` gave: `42 passed, 1 xfailed in 42.41s`.
  - Control first: slope error 5.9e-8, curvature error 8.0e-4.
  - Oracle v2 at the smallest step, bootstrap SE: **d<z>/d(rho) = 1.0015 +- 0.0015** (weighted mean over four converged rungs 1.0032 +- 0.0007), **d<z>/d(sigma) = 0.1494 +- 0.0015**, **d<z>/d(beta) = -1.658 +- 0.006** (not converged at the largest step).
  - rho stays near the Phase 2 value. It did not move to 1.012 or above.
  - Compute: about 4.4 minutes on CPU (88 s per parameter).
- Predictions against results: see the table in `docs/PREDICTIONS-A1.md`. Misses: rho description, rho ladder behaviour, sigma convergence, beta convergence, compute time.
- Caveats:
  - Our LSS (1.016) and the nilss-jax README (about 1.02) both sit about 1.3% to 1.7% above this oracle. I did not find the cause. A2 and later tasks should not assume either side is right.
  - Single averaging window (100) and dt (0.01). Not varied here.
  - The sigma prediction is not independent of the nilss-jax README numbers relayed in `TODO.md`. I read no nilss-jax source.
  - The beta convergence failure is marginal (0.0197 against a limit of 0.0165) and in the pair with the largest steps.
- Next step: owner review. Then A2 (run nilss-jax on Lorenz-63 for rho, sigma, beta). Pin one exact commit first, record the hash, and write `docs/PREDICTIONS-A2.md` before running. Do not read nilss-jax source. A decision for the owner: whether to run a window-length check on the oracle before A2, given the LSS gap.

## 2026-10-27 Task A2 session: run nilss-jax on Lorenz-63

- Status: A2 done. Controls pass. **A2.1 fails for rho and sigma, passes for beta.** A2.2 holds (the sigma claim reproduces and the gap to the oracle is confirmed). A3 not started.
- Changes: `docs/PREDICTIONS-A2.md` (predictions, pin and disclosure committed before nilss-jax was installed or run, a timing Amendment, results appended), `experiments/phaseA2_run.py`, `results/phaseA2_*.json`, `READING-LOG.md` row. Branch is stacked on `claude/phase-A1-oracle-v2` because A2 needs its oracle file.
- Pin: nilss-jax 0.1.0, commit `ad5b635f5798d89f5d31afa4118d31b778ab86f7` (2026-10-08). Cloned into the session scratchpad after the owner allowed github.com. I read only its README, CHANGELOG and `docs/usage.md`. I read no source.
- Verification:
  - `python -m pytest -q` gave: `42 passed, 1 xfailed in 36.68s`.
  - Controls first: leading exponent 0.9053 (against 0.9041), mean z 23.552 (against 23.55).
  - Stage 2 (32 orbits, T = 1000, 23 s): rho 1.01708 +- 0.00007, sigma 0.13375 +- 0.00008, beta -1.65927 +- 0.00116. Oracle (A1): 1.00317 +- 0.00074, 0.14882 +- 0.00072, -1.65814 +- 0.00595. Differences +1.39%, -10.13%, -0.07%.
- Predictions against results: all hit, but I made them after reading the README and `docs/usage.md` numbers, so they are weak evidence. Only the compute-time prediction missed.
- Caveats:
  - `docs/usage.md` showed more numbers than `TODO.md` listed, including the tool's own finite-difference values. Disclosed in the predictions file.
  - My LSS for rho (1.014 to 1.016) agrees with nilss-jax (1.017) and not with the oracle (1.003). I do not know which is right. The oracle has one window and one dt.
  - Its per-orbit scatter (0.0004 for rho) is far smaller than the gaps, so the gaps are not noise.
- Next step: owner review. Then A3 (vary `dt`, `T_seg`, `nus`, `T` and extend our own LSS to sigma and beta). A3 can use larger ensembles since a T = 1000 orbit takes 0.7 s. Consider an oracle window-length check (averaging window, dt, spin-up) before A4, since the disagreement is between shadowing and finite differences.

## 2026-10-27 Task A1b session: oracle window, dt and spin-up check

- Status: A1b done. Control passes. **The oracle does not depend on the averaging window, dt or spin-up for rho and sigma** at the resolution of this check (shifts under about 0.004). The A2 gap is therefore not explained by these three settings. Beta depends on spin-up (spin-up 10 shifts it by 5 SE). A3 not started.
- Changes: `docs/PREDICTIONS-A1b.md` (design and gates committed before any code ran, results appended), `experiments/phaseA1b_windows.py`, `results/phaseA1b_windows.json`, `.gitignore`. Branch is stacked on `claude/phase-A2-nilss-run` (which is stacked on A1).
- Verification:
  - `python -m pytest -q` gave: `42 passed, 1 xfailed in 38.71s` (no new test).
  - Control C-W: baseline matches the A1 rung within 1.25, 0.39 and 1.09 SE for rho, sigma, beta.
  - rho by variant: 0.9976 to 1.0033 (baseline 1.0004 +- 0.0015). sigma: 0.1456 to 0.1514 (baseline 0.1497 +- 0.0015). A2 shadowing values: 1.017 and 0.134.
  - Compute: 72 s on CPU.
- Predictions against results: control hit, rho and sigma stability hit, "no variant reaches shadowing values" hit. Misses: beta spin-up 10 (5.1 SE, I predicted none beyond 3 SE and beta within 0.02), compute time (72 s against 4 min).
- Caveats: one step h per parameter, one factor at a time, box start points. The check removes three possible biases in the oracle. It does not say which method is closer to the true derivative.
- Next step: owner review, then A3 (vary `dt`, `T_seg`, `nus`, `T` for nilss-jax and extend our own LSS to sigma and beta). With the oracle stable, A3's LSS for sigma and beta is the most informative next step: it can show whether an independent shadowing code also gives sigma near 0.134.
