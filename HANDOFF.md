# Handoff log

Append-only. Newest entry last. Format: Status, Changes, Verification, Caveats, Next step.

## 2026-10-26 planning session: repo seeded

- Status: plan, agent rules and an empty scaffold written. No experiment has been run and no phase 1 code exists.
- Changes: `PLAN.md`, `AGENTS.md`, `README.md`, `READING-LOG.md`, `pyproject.toml`, `src/chaos_jax/__init__.py`, `tests/test_env.py`, `reference/` submodule (chaos-reference).
- Verification: see the commit message for the scaffold test result; nothing else was run.
- Caveats: the Lorenz λ₁ value in the plan (about 0.906) comes from memory of the literature and is not confirmed in the reference set; Phase 1 measures it. Whether a cloud session can fetch the `reference/` submodule is untested. The cloud environment's Python version and whether Julia exists are unknown.
- Next step: Phase 1 (core and controls) on branch `claude/phase-1-core`.
