# chaos-jax

A small lab that measures when gradients through chaotic simulations fail in JAX, and which fixes work. Lorenz-63 first, a double pendulum second, an optional Julia port last.

Status: Phases 1 (core and controls) and 2 (oracle for Q_b) built. See HANDOFF.md. Start at [PLAN.md](PLAN.md). Rules for agents are in [AGENTS.md](AGENTS.md); progress notes in [HANDOFF.md](HANDOFF.md).

Private repository. Papers live in the separate private repository `chaos-reference`, mounted at `reference/` as a submodule.

## Set up

```bash
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Expected: one passing test (`tests/test_env.py`, checks float64 is on).
