# chaos-jax: plan

Written 2026-10-26 by the owner's planning session. The cloud session builds; the owner's planning session reviews each phase before it counts as done.

## The question, in plain words

A **chaotic** system is one where two almost identical starting points drift apart exponentially fast. The rate is the **Lyapunov exponent** (λ). If you ask autodiff for the gradient of a result with respect to a starting value or a parameter, it multiplies Jacobians along the whole trajectory, so the gradient also grows like e^(λT). Past a few 1/λ it is huge, wildly different from run to run, and useless for fitting.

Questions this repo answers by measurement:

1. How fast does the autodiff gradient blow up, and does that match λ?
2. For a long-time average (the thing people usually want), where does the naive gradient stop being usable, compared with a trustworthy reference value?
3. Which fixes work, at what cost: truncating the gradient, averaging over an ensemble, least squares shadowing (LSS)? Optionally NILSS or multiple shooting shadowing (MSS).
4. In a parameter-fitting task, how does the success rate of single-shooting fall with horizon, and does multiple shooting rescue it?
5. Do the answers transfer from Lorenz to a second chaotic system (double pendulum)?
6. Optional: does a Julia port agree with JAX, and how do ergonomics and speed compare?

A short search on 2026-10-26 found theory papers on all the fixes (see `reference/LEDGER.md`) and a JAX package (lyapax) that warns about naive gradients, but no JAX implementation of MSS and no head-to-head comparison of the fixes on Lorenz plus a pendulum. That judgement came from a few searches, not a survey. The write-up must say it plainly and must not claim novelty beyond "a small, reproducible comparison".

## Systems and quantities

- **Lorenz-63**: σ=10, β=8/3, ρ=28. Parameter of interest: ρ. Commonly quoted λ₁ is about 0.906 and is **not confirmed in the reference set**, so Phase 1 measures it two independent ways and treats the quoted number only as a sanity band.
- **Double pendulum** (Phase 6): equal masses and lengths, g=9.81. Choose a high-energy case that is chaotic and a low-energy case that is not. Parameter of interest: one of the lengths or masses; document the choice.
- **Controls (non-chaotic)**: damped harmonic oscillator, and the low-energy pendulum. Their gradients must NOT blow up.
- **Q_a**: sensitivity of the final state to the starting state, ∂x(T)/∂x₀.
- **Q_b**: sensitivity of a long-time average to a parameter, d⟨z⟩/dρ for Lorenz, where ⟨z⟩ is the time average of z after discarding a transient.
- **Control for Q_b**: a driven damped linear oscillator ẍ + cẋ + kx = F cos ωt. Its steady-state mean square, F²/(2((k−ω²)²+c²ω²)), has a closed-form derivative in ω. Every method, naive included, must agree with it. If any method fails the control, the pipeline is wrong; fix that before reading any chaotic result.

## Rules that apply to every phase

- **Float64 always.** JAX defaults to float32; `chaos_jax/__init__.py` turns on x64 and a test asserts it. Float32 is only used on purpose (Phase 3).
- **Predictions first.** Before running an experiment, write the predictions and pass/fail gates in `docs/PREDICTIONS-<phase>.md`, commit it, then run. Never edit a gate after seeing results; add a dated `Amendment` section instead and say why.
- **Controls first.** Run the non-chaotic control before the chaotic case; stop if the control fails.
- **Break it.** Each phase lists deliberate bugs. Commit first, plant each bug in a copy or working tree, confirm a test fails, restore with git. A bug no test catches is a finding to report and fix.
- **Report misses.** A failed prediction goes in the results table, not in a footnote. "No usable signal" is a valid result.
- **Fast tests, separate experiments.** `pytest` must finish in under 2 minutes on CPU. Long runs live in `experiments/` and write JSON and Markdown to `results/`. Size runs to finish in under 15 minutes each and state the compute used.
- **Fresh output.** Quote the actual pytest summary line ("N passed") in each handoff. A run with no summary line counts as broken. Delete old result files before regenerating.

## Phases

Each phase ends on its own branch `claude/phase-N-<name>`, a pull request into `main`, and a `HANDOFF.md` entry. Stop at the end of a phase; the owner reviews before the next starts.

### Phase 0: scaffold (done by the planning session)

`pyproject.toml`, `src/chaos_jax/__init__.py` with x64 on, `tests/test_env.py`, these docs. Check: `pytest` shows one passing test.

### Phase 1: core and controls

Build in `src/chaos_jax/`: an RK4 integrator using `jax.lax.scan`; vector fields for Lorenz-63, the damped oscillator and the double pendulum; tangent propagation with `jax.jvp` or `jax.jacfwd`; a Benettin-style Lyapunov exponent estimator.

Gates (set now):
- G1.1 RK4 convergence order is between 3.8 and 4.2 when dt is halved, on a smooth non-chaotic ODE.
- G1.2 Autodiff gradient agrees with a central finite difference to relative error below 1e-6 at horizon T=1, for the control and for Lorenz, in float64.
- G1.3 Lorenz λ₁ from the Benettin method, from two independent starting points, agree within 0.02 of each other and lie in [0.85, 0.95]. Record the value.
- G1.4 Damped oscillator λ is negative; low-energy pendulum |λ| < 0.02.
- G1.5 Double pendulum at the chosen high energy has λ > 0.1; relative energy drift over T=10 stays below 1e-6 at the chosen dt (record dt).

Break-it: flip the sign of the β term in Lorenz; use the wrong dt in one RK4 stage; run the whole test file in float32. Each must fail a test.

### Phase 2: ground truth for Q_b

Compute d⟨z⟩/dρ at ρ=28 by ensemble finite differences: many random initial conditions on the attractor, long averaging time, ρ±h, slope with a bootstrap standard error over initial conditions. Also check the response is close to linear over ρ in [27, 29].

Gates:
- G2.1 Standard error is below 10% of the estimate. If not, raise the ensemble or averaging time until it is, and record the cost.
- G2.2 The response is linear over [27, 29] within the error bars. If not, report it and stop: the sensitivity may not be well defined and later phases need rethinking.
- G2.3 Control: the driven-oscillator derivative from the closed form matches the same finite-difference pipeline to 1e-6.
- Sanity band: the number may be compared with any value stated in `reference/text/arxiv-1204.0159.txt` once the session has found it there. If the paper states a value, record it. Do not tune the oracle to match it.

### Phase 3: gradient blow-up

Write predictions first (`docs/PREDICTIONS-3.md`), then measure.
- P3.1 The slope of log‖∂x(T)/∂x₀‖ against T over T in [2, 20], averaged over at least 50 initial conditions, is within 15% of the Phase 1 λ₁.
- P3.2 The two control systems have slope at most 0.02.
- P3.3 The naive gradient of Q_b over horizon T (time average of z, unrolled autodiff) has median magnitude more than 10x the oracle at some T*. Write one predicted number for T* before running; report the miss.
- P3.4 Float32 gradients leave the float64 values at a smaller T than float64 leaves the oracle. Predict which T before running.

### Phase 4: the fixes

Compare, on Q_b with horizons T in {5, 10, 20, 40, 80}:
(a) naive unrolled autodiff; (b) truncated gradient (stop the gradient through all but the last few Lyapunov times); (c) ensemble average of (a) or (b) over many initial conditions; (d) LSS; (e) NILSS, stretch goal only.

Notes: LSS on T=80 at dt=0.01 has about 24,000 unknowns, so a dense solve will not fit in memory; use a sparse or block-tridiagonal solve. Read `reference/text/arxiv-1204.0159.txt` and `reference/text/arxiv-1611.00880.txt` for the formulation, and check equations against the PDF pages.

Gates: a method works at horizon T if the relative error of its median over initial conditions is at most 25% of the oracle and its relative interquartile range is at most 100%. Report cost too (wall time, peak memory). Control first: every method must reproduce the driven-oscillator derivative to 1e-4.

Break-it: drop the time-dilation term from LSS; use a shifted average window; swap the sign of the constraint. Each must be caught.

### Phase 5: fitting task

Recover ρ from a noise-free and a noisy (noise sd 0.01) Lorenz trajectory observed on [0, T_obs], starting from ρ₀ = 24 and 32. Compare single shooting with multiple shooting (window length W, penalty μ; choose W from the Lyapunov time and say how). Fixed optimiser budget. Metric: fraction of at least 50 random cases that end within 0.1 of the truth, against T_obs. Also plot the single-shooting loss against ρ at several T_obs and count local minima.

Predict first: single-shooting success falls sharply once T_obs exceeds a few Lyapunov times; multiple shooting keeps most of its success. Report what happens.

### Phase 6: second system

Repeat Phase 3 and the Phase 4 methods (a), (b), (c) on the double pendulum, with the thresholds from Phases 3-4 frozen (no retuning). The conclusion "transfers" only if the ordering of methods is the same. Say where it is not.

### Phase 7: Julia port (optional)

Only if `julia --version` works in the environment. If not, write `docs/julia-skipped.md` giving the reason, and stop. If it works: port the Phase 1 core and the Phase 3 blow-up measurement (ForwardDiff for tangents; Zygote or Enzyme optional), check gradients agree with JAX to 1e-8 at T=1, and compare timing with warm-up excluded. State Julia and package versions. `reference/` has the Zygote, Enzyme and SciML papers if needed.

### Phase 8: write-up

`RESULTS.md`: what was measured, tables with all gates and all misses, compute used, the prior-art statement above, and limits (two systems, low dimension, no PDE). No claim without a result file behind it.

## Decisions left to the owner

- Licence: none chosen yet; repo is private.
- Whether any of this goes public later (if so, the `reference` submodule pointer must be removed first).
- Whether to add Kuramoto-Sivashinsky (PDE, from `arxiv-1307.8197`) after Phase 6.
- Whether the Julia port is worth doing.
