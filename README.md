# chaos-jax

A small lab that measures when gradients through chaotic simulations fail in JAX, and which fixes work. Lorenz-63 first, a double pendulum second, an optional Julia port last.

Status: Phases 1 to 4 and audit tasks A0 to A5 are merged (main 7b61f54). The project is now an independent audit of the JAX NILSS tool `nilss-jax`, not a new method: the idea that plain gradients fail on chaos and shadowing fixes it is established (Metz et al. 2021; Chandramoorthy, Magri and Wang 2020; Lasagna and Sharma, arXiv 1806.02077, which already reports a ~2% bias of shadowing against finite differences for the Lorenz-63 sensitivity to rho). See HANDOFF.md. Start at [PLAN.md](PLAN.md). Rules for agents are in [AGENTS.md](AGENTS.md); progress notes in [HANDOFF.md](HANDOFF.md).

Licence: MIT (see `LICENSE`).

## What is in this repository

- **chaos-jax (this directory):** an independent audit of the JAX NILSS package `nilss-jax` on Lorenz-63, with a finite-difference oracle. Read [RESULTS.md](RESULTS.md) first. It is a replication and audit, not a new method.
- **[hyperchaos/](hyperchaos/):** an evaluator for chaotic and hyperchaotic flows (Lyapunov spectrum, controls, planted-bug checks) and a catalogue of 3D quadratic chaotic flows it verified: 5,101 hits in 1,260 structure families, in [hyperchaos/catalogue/](hyperchaos/catalogue/). "Chaotic" means numerical evidence under the stated checks. Most families are probably rediscoveries; only Sprott's published cases were compared with the literature. It also records negative results (no dissipative chaotic flow with four terms; no verified hyperchaotic 4D flow among 369,600 four-term systems; two published 4D hyperchaos claims that did not reproduce at their stated settings). Start at [hyperchaos/README.md](hyperchaos/README.md).

## Reference papers

The papers used for reading are not in this repository (some are paywalled and are not ours to redistribute). Documents here cite them by arXiv id or title; `reference/...` paths in older notes (`docs/`, `HANDOFF.md`, `READING-LOG.md`, `AGENTS.md`) point at a private paper library that is not part of this repository.

## Set up

```bash
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Expected: one passing test (`tests/test_env.py`, checks float64 is on).
