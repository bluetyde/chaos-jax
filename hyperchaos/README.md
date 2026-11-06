# hyperchaos-catalogue

A verified catalogue of the smallest chaotic and hyperchaotic flows, found by a search with a strict evaluator.

Status (2026-11-06): stages H0 to H3 done, with a 5- and 6-term 3D search and a four-term 4D search. H4 (clean up), H5 (novelty check) and H6 (write-up) are not done. Start at `catalogue/` for the data, then `PLAN.md`, `TODO.md`, `HANDOFF.md`, `LITERATURE.md`.

## Results so far

- **Catalogue:** `catalogue/families.csv` (one row per structure family, 1,260), `catalogue/hits.csv` (all 5,101 kept hits) and `catalogue/catalogue.json`. Each row has the equations, term count, three Lyapunov exponents with standard errors, Kaplan-Yorke dimension and the coefficient vector. Regenerate with `python experiments/export_catalogue.py`. These are 3D flows with quadratic right-hand sides found in random searches and verified by the evaluator; hits that H1 Amendment 5 removes (positive exponent sum) are excluded. Families are grouped by renaming, sign flips and positive rescaling, not matched against the literature except Sprott's A to S.
- **Negative results:** no dissipative chaotic flow among exhaustive four-term 3D systems (rank-4 and rank-3 supports; rank-2 coarse); no verified chaotic or hyperchaotic flow among 369,600 four-term 4D systems; Kopp 2025 hyperchaos does not reproduce (second exponent at most 0.0004); Benkouider 2024 second exponent not reproduced (0.005 or less against 0.010 published); some stiff-step false positives found and removed.
- **Not claimed:** no record, no "simplest known", no new system.

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
