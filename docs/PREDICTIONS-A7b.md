# Task A7b predictions and gates: `fds` as a black box on Lorenz-63

Written and committed before any `fds` run. Do not edit a gate or prediction after results. Add a dated Amendment instead.

## What I know about `fds`, and how

- Package: `fds` on PyPI, release `0.1.0.dev1` (uploaded 2016-09-28, the newest of three pre-releases). Licence GPL-3.0. Python 2/3 wheel. It needs `mpi4py`. I installed it into a scratch virtual environment outside this repository, with `mpich` (a PyPI package) to give `mpi4py` its MPI library. Exact versions of every package go into the results file.
- Documentation read: the home page and the Van der Pol tutorial at `fds.readthedocs.io` (a 0.0.2 build). The pages "The least squares shadowing algorithm", "Function reference", the three guides and the two other tutorials are **empty** on that site (261 characters of page frame, no content). So I could **not** read which formulation `fds` implements. The home page says only that the algorithm is in "the upcoming AIAA paper", and I do not have it.
- Interface: the tutorial shows `J, G = shadowing(run, u0, parameter, subspace_dimension, num_segments, steps_per_segment, runup_steps)`, with `run(u0, parameter, nsteps)` returning the final state and the objective time series. I also printed `inspect.signature(fds.shadowing)` (names and defaults only, no docstring and no function body). This is the only thing I looked at inside the package. It shows an extra argument `epsilon=1e-6`. I did not read any source.
- Because the formulation is unknown, I do **not** know whether `fds` is an LSS code, a NILSS code or something else. It takes my solver as a function, so its tangent information must come from runs of that function (the name `epsilon` suggests finite differences of runs). That is a guess.

## What I have seen of the other numbers (so my predictions are not independent of them)

- nilss-jax (A2): rho 1.01708, sigma 0.13375, beta -1.65927. Own LSS (A3) and own NILSS (A5): within 0.5% of those.
- Oracle (A1, weighted mean over rungs): rho 1.0032 +- 0.0007, sigma 0.1488 +- 0.0007. For beta I use the finest rung, -1.658 +- 0.006 (the rungs do not all agree for beta).
- The planning session's own numpy oracle reproduced the oracle.

## Method (fixed now)

Lorenz-63 with (sigma, rho, beta) = (10, 28, 8/3), objective J = z. `fds` takes one scalar parameter, so I run three separate `shadowing` calls per orbit, one for each of rho, sigma and beta. Each call uses a wrapper `run(u0, parameter, nsteps)` that integrates the Lorenz system with the other two parameters fixed.

- Wrapper 1 (main): RK4 in plain numpy at dt = 0.005, the same scheme as my JAX code. Code in `experiments/fds_lorenz.py`.
- Wrapper 2 (second integrator): `scipy.integrate.solve_ivp` with DOP853 and `rtol = atol = 1e-12`, sampled every dt = 0.005. Same file.
- Settings: `subspace_dimension = 1`, `steps_per_segment = 100` (0.5 time units, as nilss-jax and my NILSS), `runup_steps = 4000` (20 time units), `epsilon` at its default. `num_segments = 2000` (T = 1000), as in A5. If the first probe shows that this is too slow, I use the fallback below.
- Start points: the 32 A2 points (`PRNGKey(70)`, as in A5), saved in `results/phaseA7b_x0s.npy`.
- Probe first: time one orbit at T = 100 for one parameter, then state the estimate for the main run. Fallback rule written now: if the projected total for 16 orbits and three parameters is above 60 minutes, use 8 orbits (the first 8 start points). I record the choice and the reason in the results file before the main run. I run one heavy job at a time.
- Main run: RK4 wrapper, N orbits, three parameters.
- Integrator run: DOP853 wrapper, the first 8 orbits at T = 500 (if the same segment count is too slow I say so), three parameters.

## Controls first (before any main sensitivity is read)

- **C-F1 wrapper against JAX (unit test, no `fds`):** the numpy RK4 wrapper gives the same state and z series as `chaos_jax.integrate` after 1000 steps, to relative error 1e-9, for the default parameters and for each of rho, sigma and beta shifted by 1e-3. The DOP853 wrapper matches the same JAX trajectory to 1e-6 after 200 steps.
- **C-F2 known case (needs `fds`):** the Van der Pol tutorial case (mu = 1, J = x squared, `odeint`, as on the docs page) gives a derivative G within 10% of a central finite difference of the long-run mean x squared over mu = 0.95 and 1.05 (200 000 steps of 0.01 each). The orbit is periodic, so this only checks the wiring and the scale of `fds`, not chaos. I chose 10% because a finite-length mean of a periodic orbit has an error of about the period divided by the length.
- **C-F3 Lorenz sanity:** the J that `fds` returns on the main run has a mean over orbits within 0.2 of 23.55 (the A1 and A5 value).

If any control fails, I stop and report. I do not read the sensitivities as results.

## Gates (after the controls pass)

All gates use the ensemble mean over orbits and its standard error (SEM = SD over orbits divided by the square root of N).

- **G-F1 (practical):** for each of rho, sigma and beta, the `fds` RK4 mean is within 1% of the nilss-jax mean from A2. Reported next to it: the difference to the oracle, with the same percentage.
- **G-F2 (which side):** for rho and sigma only. Let S be the nilss-jax mean and O the oracle mean. If `fds` is closer to S than to O, I say "shadowing side". If closer to O, "oracle side". If it is farther than 1% from both for sigma, or farther than 0.5% from both for rho, I say "neither". Beta cannot discriminate (S and O agree) and I say so.
- **G-F3 (integrator):** for each parameter, the DOP853 mean agrees with the RK4 mean on the same orbits within 2 combined standard errors. If the DOP853 run fails or is too noisy to compare, I report that as the result and do not tune `epsilon` or the tolerances after seeing it. A second try with a different `epsilon` needs a dated Amendment first.
- Reported, not gated: wall time per orbit for both wrappers; whether `fds` runs at all with the adaptive solver.

## Break-it (after committing the code)

Plant each bug in the working tree, confirm a test or control fails, restore with git.
1. Swap the roles of sigma and rho in the wrapper. Must be caught by C-F1.
2. Return x instead of z as the objective. Must be caught by C-F1 (series) and C-F3.
3. Use dt = 0.005 in the integrator but report the series at every second step. Must be caught by C-F1.
A bug that nothing catches is reported, and the tests are fixed.

## Predictions (not gates)

- **P-F1** `fds` installs (done) and runs on Python 3.13 with NumPy 2.x. I give this about 60%, because the package is nine years old. If it fails to run, I write `docs/fds-skipped.md` and stop.
- **P-F2** C-F1 to C-F3 pass at the first run, except that I may repair my own wrapper (a repair counts as a miss).
- **P-F3** `fds` lands on the shadowing side for rho and sigma (G-F1 passes for all three), about 65%. Reason: any shadowing algorithm gives about 1.017 and 0.134, and the `fds` authors wrote the original LSS and NILSS papers. I cannot say more, because the formulation is not documented.
- **P-F4** the sigma and rho values from `fds` have a larger spread over orbits than my NILSS, because of the finite-difference tangents. I expect an SD per orbit of 1% to 5% of the mean for sigma.
- **P-F5** the DOP853 run agrees with the RK4 run (G-F3 passes for all three), about 70%.
- **P-F6** wall time per orbit for the RK4 wrapper at T = 1000: 1 to 10 minutes. This is a guess until I run the probe.
- **P-F7** if `fds` is on the shadowing side with the adaptive solver as well, then a shared RK4 effect is not the cause of the rho and sigma gaps. If it is on the oracle side, the gap depends on the algorithm, and I would say that the three shadowing codes share an effect that `fds` does not.

## Results (written after the runs; the gates above did not change)

Versions: `fds` 0.1.0.dev1, numpy 2.5.4, scipy 1.18.1, mpi4py 4.1.2, mpich 5.0.2, dill 0.4.1, Python 3.13. Files: `results/phaseA7b_control.json`, `phaseA7b_probe.json`, `phaseA7b_main.json`, `phaseA7b_dop853.json`, `phaseA7b_report_main.json`, `phaseA7b_report_dop853.json`, `phaseA7b_gf3.json`.

**Controls.** C-F1 (9 tests, `tests/test_fds_wrapper.py`) passed at the first run. C-F2 (Van der Pol): `fds` G = 0.1136, central difference 0.1118, difference 1.6%, gate 10%, pass. C-F3: mean J 23.550 (main) and 23.538 (DOP853), gate 23.55 +- 0.2, pass.

**Run choices written before the main run.** The probe took 22 s for T = 100, so 16 orbits would have taken about 2.9 hours. The written fallback rule gave 8 orbits. The real time was longer than the projection: about 290 to 300 s per call at T = 1000 (about 15 minutes per orbit for the three parameters), so the main run took about 1 hour 50 minutes. The DOP853 run used 8 orbits at T = 500 and took about 144 s per call, about 1 hour in all.

**Main run (RK4 wrapper, 8 orbits, T = 1000).** Mean +- SEM over orbits.

| Parameter | `fds` | nilss-jax (A2) | Oracle (A1) | `fds` against nilss-jax | `fds` against oracle | G-F1 | G-F2 |
|---|---|---|---|---|---|---|---|
| rho | 1.01709 +- 0.00024 | 1.01708 | 1.00317 | +0.0007% | +1.39% | pass | shadowing side |
| sigma | 0.13362 +- 0.00021 | 0.13375 | 0.14882 | -0.10% | -10.22% | pass | shadowing side |
| beta | -1.65926 +- 0.00239 | -1.65927 | -1.65814 | +0.0008% | -0.07% | pass | not decisive |

**Second integrator (DOP853, rtol = atol = 1e-12, same 8 orbits, T = 500), G-F3.**

| Parameter | RK4, T = 1000 | DOP853, T = 500 | Difference | Combined SE | Result |
|---|---|---|---|---|---|
| rho | 1.01709 | 1.01714 | +0.00006 | 0.00036 | pass (0.16 SE) |
| sigma | 0.13362 | 0.13386 | +0.00025 | 0.00032 | pass (0.78 SE) |
| beta | -1.65926 | -1.66383 | -0.00458 | 0.00323 | pass (1.42 SE) |

The adaptive solver ran with `fds` without any change of `epsilon` or tolerances. It sits on the shadowing side too: rho +1.39% and sigma -10.05% from the oracle.

**Break-it (run after the code commit).** Bug 1 (sigma and rho swapped): 9 of 9 tests failed. Bug 2 (x returned in place of z): 6 failed, 3 passed (the three DOP853 tests, which the bug did not touch). Bug 3 (series reported at every second step): 6 failed, 3 passed (same reason). All three were caught. Files restored with git.

**Predictions against results.**

| Prediction | Outcome |
|---|---|
| P-F1 `fds` runs on Python 3.13 and NumPy 2.x (60%) | **Hit.** It needed `mpich` from PyPI for the MPI library, and a picklable `run` object (see below). |
| P-F2 controls pass at the first run, wrapper repairs count as a miss | **Miss.** C-F1 to C-F3 passed, but my first control call failed: `fds` sends `run` to worker processes, so a lambda cannot be used. I replaced the lambdas with a picklable class. This was a mistake in my script, not a failure of `fds`. |
| P-F3 shadowing side for rho and sigma, G-F1 passes for all three (65%) | **Hit.** |
| P-F4 spread over orbits 1% to 5% of the mean for sigma, larger than my NILSS | **Miss.** The spread is 0.44% for sigma and 0.07% for rho, about the same as my NILSS. |
| P-F5 DOP853 agrees with RK4 (70%) | **Hit.** |
| P-F6 1 to 10 minutes per orbit at T = 1000 | **Partly.** About 5 minutes for one parameter (in range), about 15 minutes for the three parameters of one orbit (out of range). The projection from the probe was also low by about 30%. |
| P-F7 conditional reading | The condition "`fds` on the shadowing side with the adaptive solver" holds. |

**What this shows, and what it does not.**
- A fourth code, written by other people and run as a black box, with RK4 and with an adaptive solver, gives the same rho and sigma as nilss-jax and my two codes. So a shared RK4 or quadrature effect in my three codes is not the cause of the +1.4% (rho) and -10% (sigma) gaps to the finite-difference oracle.
- It does not tell which side is nearer to the true derivative. All four codes are shadowing-type methods, so they can share a method-level effect (for example a missing unstable contribution, or non-hyperbolicity).
- I could not read which formulation `fds` implements (the documentation page is empty and I did not read the source). I only know that it has a subspace dimension, segments, a run-up and a finite-difference step `epsilon`, so it looks like a NILSS-type code. That is a guess.
- Limits: 8 orbits only, one version of `fds` (2016, a pre-release), one objective, one set of settings, T = 500 for the adaptive run against T = 1000 for the RK4 run, and finite-difference tangents with `epsilon` = 1e-6 that I did not vary.
- Source of `fds` is still unread. The GPL-3.0 licence note stands: no source was copied into this repository.
