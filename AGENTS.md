# Agent instructions

Read `PLAN.md`, then `TODO.md` (the current task list and direction), then `HANDOFF.md` (newest entry last), then `READING-LOG.md`, before changing anything.

## Scope and flow

- One phase per session. Stop at the end of the phase and say what is next. Do not start the next phase.
- Work on a branch named `claude/phase-N-<name>`. Never push to `main`. Open a pull request and leave merging to the owner.
- Append an entry to `HANDOFF.md` before finishing: Status, Changes, Verification (exact commands and the real output lines), Caveats, Next step.

## Evidence rules

- Do not write "verified" or "passes" without a command whose output you quote. A pytest run with no "N passed" summary line is a failure.
- Predictions and gates are written and committed before the experiment runs. Never change a gate after seeing results; add a dated Amendment.
- Run controls before the chaotic case. If a control fails, fix the pipeline first.
- Plant the listed deliberate bugs after committing, confirm tests fail, restore with git. Report any bug no test caught.
- Report failed predictions in the results table. "No usable signal" is an acceptable outcome.
- Float64 only, unless a phase says otherwise.
- Run one heavy job at a time, and say how long it should take before you start it.

## Reference papers

- The papers live in a separate private repository, `bluetyde/chaos-reference`, mounted at `reference/` as a git submodule. If `reference/` is empty, run `git submodule update --init`. If that fails (no access), say so in `HANDOFF.md` and carry on with the parts of the plan that need no paper.
- Search `reference/text/*.txt` first. Check every equation you implement against the PDF page; extracted text garbles symbols.
- Record in `READING-LOG.md` what you read, how much (full, section, or search only), and what you took from it. Never describe a paper you did not open.
- Reimplement methods from the mathematics. Do not copy code from the authors' repositories, and never copy PDFs or long passages into this repo.
- If a needed paper is listed in `reference/NEEDED-FROM-OWNER.md` and not present, say so and skip that line of work.

## Hygiene

- Git identity in this repo: user.name `bluetyde`, user.email `230244426+bluetyde@users.noreply.github.com`. Set it before the first commit. Never commit with any other email.
- No personal names, emails, Windows paths or machine details in any file.
- LF line endings. Python 3.10 or newer. Tests under `tests/`, long runs under `experiments/` writing to `results/`.
