# Handoff log (newest entry last)

## 2026-10-26: folder seeded (Claude Code, Windows, planning session)
- Status: new local folder `hyperchaos-catalogue` inside the owner's `Claude Code Test` workspace, local git repo on `main`, no remote. No code run except the scaffold smoke test.
- Changes: README, PLAN (H0 to H6), AGENTS, TODO, LITERATURE (leads from 13 web searches and 6 GitHub searches, abstracts only), scaffold (`src/hyperchaos`, one smoke test).
- Verification: `python -m pytest -q` in the folder, using the chaos-jax venv (Python 3.14, Windows): 1 passed in 1.44s. Planted bug (x64 line removed from `__init__.py`): 1 failed, then restored: 1 passed. No CR characters in the text files (grep count 0).
- Caveats: the hyperchaos literature has NOT been read; every row in `LITERATURE.md` is a lead. Rössler's equations in that file are from memory. The whole project depends on the H0 table being filled honestly before any "record" language. Prior-art notes and the reasoning for this project are in `Claude Code Test\messages\ideas.md` (Moonshot 7).
- Next: H0, tasks H0.1 to H0.8 in `TODO.md`, on the owner's word.
