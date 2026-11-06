# Agent instructions

Read, in this order: `PLAN.md`, `TODO.md`, `HANDOFF.md` (newest entry first), `LITERATURE.md`.

## Working rules

- One stage per branch, named `claude/h<N>-<name>`. Do not start a stage the owner has not approved.
- Write predictions and gates in `docs/PREDICTIONS-H<N>.md` and commit them BEFORE the first run. Gates are not changed after a run; a miss is reported as a miss.
- Run the controls first. A result without its control is not a result.
- A stage with code ends with a planted-bug check: break the code on purpose and show a test fails, then restore.
- Python 3.10 or newer, JAX with `jax_enable_x64` on. Everything numerical is float64. Never use float32 for an exponent.
- Judge a run by its exit code AND its summary line. An empty or missing test summary is a failure.
- Delete an old results file before the run that should replace it, and confirm the new one is fresh.

## Evidence rules

- Call something chaotic or hyperchaotic only if it passed every H1 check, and say "numerically". Never say "proved".
- Never write "record", "simplest known", or "new" until H0 and H5 are filled in for that claim. Say what was searched and what was not.
- Say what you read in full and what you only saw in a search result or abstract. Do not quote equations or term counts from memory; copy them from the paper and cite it.
- Report every miss against the predictions, including the ones that look embarrassing.
- Independence: do not copy source code from other chaos or Lyapunov libraries. Ideas are fine. If you read a library's source, say so in `HANDOFF.md`. Cross-check against `lyapax` or `jitcode` as black boxes with pinned versions.

## Hygiene

- Git identity: user `bluetyde`, email `230244426+bluetyde@users.noreply.github.com`. Check `git config user.email` before the first commit. Scan staged files for the owner's real name, personal email, and Windows paths before any push.
- Do not push, publish, post to issues, or create a remote repo without the owner's explicit word each time.
- Unix LF line endings in every file.
- Papers with unclear licences stay out of this repo; list them in `LITERATURE.md` and ask the owner for copies.
- Commit trailer on every commit: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- End every session with a `HANDOFF.md` entry: status, changes, exact verification commands and results, caveats.
