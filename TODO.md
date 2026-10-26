# TODO: next tasks

Written 2026-10-26 by the planning session at the owner's request. Where this file and a "Next step" line in `HANDOFF.md` differ, this file wins. Decisions are marked **owner** (the owner made it) or **planning** (recommended by the planning session; the owner can overrule).

## Where things stand

Phases 1 to 4 are merged (PRs 1 to 4). Phase 2 oracle for d<z>/d(rho) at rho = 28: 1.0029 +- 0.0021 (ensemble finite differences, central difference over rho = 27 and 29). Phase 4: our LSS gives 1.016 at T = 80 with a very small spread; the LSS control gate is a strict xfail (error 4.5e-2 against 1e-4, falling like 1/T).

## What changed, read this before planning anything

1. **A JAX implementation of NILSS already exists:** `BrianP2002/Guiding_Center_Optimizer` on GitHub (installs as `nilss-jax`; MIT; one author; first release v0.1.0 dated 2026-10-07 to 08; 7 commits; 0 stars). The planning session read its README, changelog and the headings of `docs/findings.md`. It did not run or audit the code. Its README claims: streaming multi-parameter NILSS in JAX; a dense reference solver that matches the streaming code to 1e-10; a finite-difference ensemble reference; a Lyapunov-spectrum tool; Lorenz-63, Lorenz-96 and Roessler examples; a "reliability report" (verdicts OK, CAUTION, UNRELIABLE) based on heavy tails of the shadowing direction and other checks. It states that on Lorenz-63 its d<z>/d(rho) is "near 1.02", the rho and beta sensitivities are within 1.5% of finite differences, and the **sigma sensitivity is about 11% off** (+0.133 against +0.150 +- 0.004 from finite differences). Treat all of this as unverified until reproduced here.
2. **Julia already ships these methods:** SciMLSensitivity has `ForwardLSS`, `AdjointLSS`, `NILSS` and `NILSAS` with a Lorenz tutorial.
3. **So the novelty claim shrinks** to "a small, reproducible, independent comparison". **Owner decision (2026-10-26): the project's direction is now an independent audit of nilss-jax**, using our finite-difference oracle. Phase 3 stays as done. Phases 5 and 6 become optional. Phase 9 stays proposed and comes after the audit.
4. **An open observation:** our LSS (1.016) and the nilss-jax README value (about 1.02) both sit above the oracle (1.0029). The Phase 2 follow-up quadratic fit over rho 25 to 50 gave 1.0101 +- 0.0005 at rho = 28. We do not know which estimate is nearest the truth. Do not assume the oracle is right.

## Rules for this audit

- **Independence.** Do not read nilss-jax's source code (`src/`) until your own implementation (task A5) exists and its results are logged. Reading its README and docs is fine; log what you read in `READING-LOG.md`.
- **Pin it.** Install nilss-jax at one exact commit and record the hash in every results file. Do not follow its main branch.
- **Disclose what you have seen.** The planning session has already read the numbers in item 1. Say so in each predictions file, as `docs/PREDICTIONS-2.md` did for the paper's figure, and do not tune anything to them.
- **All earlier rules hold:** predictions and gates committed before any run, controls first, no gate changed after seeing results (add a dated Amendment), planted-bug checks, one heavy job at a time, state expected run time before starting.
- **Post nothing** to the nilss-jax repository or anywhere else public. The owner decides that.
- **One task per branch** named `claude/phase-<id>-<name>`, one pull request, one `HANDOFF.md` entry, then stop.

## Tasks, in order

**A0 (small): tidy Phase 4.** Add a dated Amendment to `docs/PREDICTIONS-4.md` saying the LSS control gate cannot be met on a 10-period window and why (the error falls roughly like 1/T). Keep the xfail and do not weaken the gate. You may add a separate, clearly labelled long-window control test, but it must not replace the gate.

**A1: oracle v2 (owner decision: yes, and first).** Write predictions first. Re-estimate d<z>/d(rho) at rho = 28 with smaller central-difference steps (suggest h = 1, 0.5, 0.25, 0.125; raise the number of start points as h shrinks, because the noise grows like 1/h) and report each with a bootstrap standard error and the curvature. Do the same for sigma and beta around (10, 8/3): the audit needs all three. Gates to write before running: the estimates at successive smaller h agree within two standard errors (converged), or you report non-convergence plainly. State the expected run time. If the converged value for rho moves to 1.012 or above, say so; if it stays near 1.003, say that.

**A2: run nilss-jax on Lorenz-63 for rho, sigma and beta.** Use an ensemble of orbits (its own studies used 8; choose more if the error bars need it), its documented settings first. Compare with the A1 oracle per parameter. Gate set before running: agreement within two combined standard errors; report the relative difference either way. Check the claimed sigma gap (+0.133 against +0.150) directly.

**A3: how much do the settings matter.** Vary `dt`, `T_seg`, `nus` and `T` and report the sensitivity estimates. The README says the sigma gap is stable against them; test that. Extend our own LSS to sigma and beta as a second opinion (the parameter derivative can come from `jax.jvp`).

**A4: calibrate the reliability report.** Over a set of cases (Lorenz-63 at several parameter values, Lorenz-96 at several sizes, Roessler), tabulate the report's verdict against the true error from a finite-difference oracle for the same case. Report false alarms (UNRELIABLE with small error) and misses (OK with large error). Gate to write before running, for example: no OK verdict with an error above 25%. Estimate the cost first; the oracle for the larger systems may be expensive, so say what you drop.

**A5 (optional): our own NILSS.** Read `reference/text/arxiv-1611.00880.txt` and `arxiv-1711.06633.txt` first and log it. Implement NILSS from the papers without reading nilss-jax source. Compare with nilss-jax at the same settings. Check equations against the PDF pages; remember Phase 4 found four printed equations in the LSS paper that disagree with each other.

**A6: double pendulum.** Not bundled in nilss-jax. Choose a bounded long-time average and one parameter. Check first that the long-time average does not depend on the starting point (the spread across starts). A Hamiltonian flow with a mixed phase space may fail this; if so, say so and stop this task.

**A7 (optional, only if `julia --version` works in the environment): Julia cross-check.** Run SciMLSensitivity's `NILSS` on Lorenz-63 and compare with the oracle and nilss-jax. If Julia is missing, write `docs/julia-skipped.md` and move on.

**A8: write-up.** `RESULTS.md`: what agreed, what did not, with numbers, error bars and run times; every failed prediction; the prior art named plainly (nilss-jax, SciMLSensitivity, Lea et al. 2000, Eyink et al. 2004, the truncation and warm-up literature); the limits (few systems, one author's young code). Describe it as an independent audit and replication. Claim nothing as new that is not.

## Still open (owner)

- Whether to post any finding upstream.
- Whether Phases 5, 6 and 9 go ahead after the audit.
- Whether the Julia work is worth it.
- Licence, and whether any of this goes public (if so, remove the `reference/` submodule pointer first).
