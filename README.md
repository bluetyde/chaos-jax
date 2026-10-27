# hyperchaos-catalogue

A verified catalogue of the smallest chaotic and hyperchaotic flows, found by a search with a strict evaluator.

Status: seeded 2026-10-26, nothing run yet. Start at `PLAN.md`, then `TODO.md`, `HANDOFF.md`, `LITERATURE.md`.

## What this is

- **Chaotic flow:** a deterministic set of differential equations whose motion is bounded, never settles or repeats, and has a positive largest Lyapunov exponent.
- **Hyperchaotic flow:** two or more positive Lyapunov exponents. A continuous-time hyperchaotic system needs at least four variables (4D).
- **Size:** the number of terms in the right-hand sides, and how many of them are nonlinear.
- **The search:** try many small systems, screen them with a strict evaluator, and keep the best system for each size in a quality-diversity map (best per cell, cells by number of terms and number of nonlinear terms).

## What this is not

- Not a proof. Every "chaotic" or "hyperchaotic" label here means numerical evidence under stated checks.
- Not a record hunt. 3D "simplest chaotic flow" is a closed question (Sprott, 1994 to 2000). Hyperchaos in 4D and up is crowded with hand-built papers. A new smaller system is possible but not expected; the verified map and the evaluator are the sure deliverables.
- Not nuclear, and not tied to any other repo. It is independent of `chaos-jax` (it may copy the integrator and Lyapunov ideas, with the source named).

## Layout

| Path | Purpose |
|---|---|
| `PLAN.md` | Stages H0 to H6 with gates and what is predicted before each run |
| `TODO.md` | Task list, one stage at a time |
| `HANDOFF.md` | Dated log of every session (read newest entry first) |
| `LITERATURE.md` | Published systems and records, with what was read versus only seen in a search |
| `AGENTS.md` | Rules for any agent working here |
| `src/hyperchaos/` | Library (empty scaffold) |
| `tests/` | Tests |
| `experiments/`, `results/`, `docs/` | Run scripts, outputs, and prediction files |
