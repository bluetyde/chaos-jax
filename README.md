# chaos-jax

A small lab that measures when gradients through chaotic simulations fail in JAX, and which fixes work. Lorenz-63 first, a double pendulum second, an optional Julia port last.

Status: Phases 1 to 4 and audit tasks A0 to A5 are merged (main 7b61f54). The project is now an independent audit of the JAX NILSS tool `nilss-jax`, not a new method: the idea that plain gradients fail on chaos and shadowing fixes it is established (Metz et al. 2021; Chandramoorthy, Magri and Wang 2020; Lasagna and Sharma, arXiv 1806.02077, which already reports a ~2% bias of shadowing against finite differences for the Lorenz-63 sensitivity to rho). See HANDOFF.md. Start at [PLAN.md](PLAN.md). Rules for agents are in [AGENTS.md](AGENTS.md); progress notes in [HANDOFF.md](HANDOFF.md).

Private repository. Papers live in the separate private repository `chaos-reference`, mounted at `reference/` as a submodule.

## Set up

```bash
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Expected: one passing test (`tests/test_env.py`, checks float64 is on).
