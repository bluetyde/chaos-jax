# Plan

Goal: a verified catalogue and quality-diversity map of small chaotic and hyperchaotic flows, plus a trustworthy evaluator for them. A claim of a new or smaller system is a bonus that needs the literature check in H5.

Working method (same as the owner's other projects): one stage per branch `claude/h<N>-<name>`, predictions written and committed before the run (`docs/PREDICTIONS-H<N>.md`), controls first, gates fixed in advance, misses reported, a planted-bug check where a stage has code. The owner reviews each stage before the next starts.

## Definitions (fixed here; change only with a logged reason)

- **State:** `x' = f(x)`, autonomous, polynomial right-hand side, dimension 3 or 4 (later 5).
- **Size:** `terms` = number of nonzero monomial coefficients over all equations, constants included. `nonlinear` = number of those monomials of degree 2 or higher. Report both.
- **Chaotic (numerical):** bounded, not converging to a point or a loop, largest Lyapunov exponent above a threshold with its standard error clearly above zero, and the verification checks of H1 all agree.
- **Hyperchaotic (numerical):** chaotic, and the second exponent is also clearly above zero. Expected spectrum shape in 4D: `(+, +, 0, -)`.
- **Dissipative:** the exponents sum to a negative number (needed for an attractor).
- **Equivalence (H4):** two systems are the same if one becomes the other by renaming variables, flipping the sign of variables, or rescaling variables and time. Record the canonical form.

## Stages

### H0. Literature table (no code)
- Read the key papers themselves, not only abstracts: Rössler 1979 (Phys. Lett. A 71, 155), Sprott 1994 and 1997, Sprott and Linz 2000, Chlouverakis and Sprott 2006 (hyperjerk), Li and Sprott 4D simplified Lorenz, the single-nonlinear-term 4D hyperchaos papers, Kopp 2025, and the thesis of Sahle (2005).
- Fill the table in `LITERATURE.md`: system, equations, terms, nonlinear terms, exponents as published, source, "read in full" or "abstract only".
- Gate: a table of the smallest known 4D hyperchaotic flows with term counts taken from the equations themselves. Without it the word "record" is not allowed anywhere.
- Owner-supplied papers go in a separate reference repo or folder, not committed here if the licence is unclear.

### H1. The evaluator
- float64 integration (fixed-step RK4 plus an adaptive scipy DOP853 for cross-checks), Lyapunov spectrum by the Benettin QR method, classification into: diverges, fixed point, periodic, quasi-periodic, chaotic, hyperchaotic.
- Verification checks every candidate must pass: second integrator, half step, longer run with the exponents converged, several initial conditions on the attractor, standard errors from independent segments.
- Positive controls: Lorenz (largest exponent about 0.906, from the chaos-jax measurement), Rössler 3D, Rössler 1979 hyperchaos (two positive exponents), a few Sprott 3D cases once their equations are read in H0.
- Negative controls: planted periodic, quasi-periodic (two incommensurate oscillators), decaying, and diverging systems must all be rejected.
- Independent cross-check: `lyapax` (JAX) and, if easy, `jitcode` (Python, C-compiled); pin the versions; record any disagreement.
- Gate: every control classified correctly, cross-checks agree within the stated standard errors, and a planted bug in the evaluator (for example, no QR re-orthogonalisation) makes at least one test fail.

### H2. Rediscover the 3D flows
- Brute-force small 3D polynomial systems on a coarse coefficient grid, as Sprott did.
- Gate: it re-finds Sprott's published cases (list from H0). Also report the hit rate and the size of the space, which sets the budget for H3.

### H3. The 4D search
- Search space, in this order: (a) 4th-order hyperjerk `x'''' = f(x, x', x'', x''')` with few terms (Chlouverakis and Sprott did this for chaos at n=4, 5 and found some hyperchaotic cases); (b) general sparse quadratic 4D systems.
- Method: MAP-Elites (library `pyribs` or a small own implementation) with axes `terms` by `nonlinear`; fitness the second Lyapunov exponent (clipped, with a penalty for non-dissipative systems). Mutations add, remove, or perturb a term. Two-stage screening: a cheap short run, then the full H1 checks on survivors. JAX `vmap` for the cheap stage.
- Gate: it re-finds known hyperchaotic systems in the cells where the H0 table says they should be, and an empty cell is not reported as "no system exists".

### H4. Clean up the finds
- Merge equivalent systems (canonical form), perturb coefficients to test robustness, rerun survivors in independent code.

### H5. Novelty check (manual)
- Compare each surviving system against dysts, Sprott's lists, and the H0 table. Claim only what the comparison supports, and name what was searched and what was not.

### H6. Write-up
- `RESULTS.md`: the map, the evaluator and its controls, every miss against the written predictions, and the limits (numerical evidence, finite coefficient grids, search budget).

## Not planned, possible later
- 5D and above; delay systems; hidden-attractor structure (see the `hidden-attractor-search` repo by Elisov); a rigorous proof for one system with interval tools (CAPD, as Wilczak, Serrano and Barrio did for Rössler 1979).
