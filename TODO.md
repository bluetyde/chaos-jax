# TODO

Stage order is H0 to H6 (see `PLAN.md`). Do one stage at a time and wait for the owner's approval between stages.

## H0. Literature table (next)
- [ ] H0.1 Read Rössler 1979 (Phys. Lett. A 71, 155) in full. Copy the equations and the standard parameters. Fill the `LITERATURE.md` row; mark the "from memory" equations there as verified or wrong.
- [ ] H0.2 Read Chlouverakis and Sprott 2006 (hyperjerk, Chaos Solitons Fractals 28, 739). Copy every n=4 and n=5 chaotic and hyperchaotic case with its equation and term count.
- [ ] H0.3 Read Sprott 1994 and the Sprott and Linz 2000 review. Copy the list of 3D cases (A to S) with equations.
- [ ] H0.4 Find and read the single-nonlinear-term 4D hyperchaos papers (Acta Phys. Sin. 58, 5201 and the 2010 J. Comput. Appl. paper). Get equations and term counts.
- [ ] H0.5 Read Kopp 2025 and Li and Sprott's 4D simplified Lorenz paper. Same rows.
- [ ] H0.6 Check Sahle's 2005 thesis (Tübingen) and Wilczak, Serrano and Barrio for the Rössler hyperchaos proof. Note what each shows.
- [ ] H0.7 Final table: smallest known 4D hyperchaotic flows by `terms` and by `nonlinear`, from the equations themselves. State the uncertainty (papers not found).
- [ ] H0.8 List the papers the owner must supply (paywalled) in `NEEDED-FROM-OWNER.md`.

## H1. Evaluator
- [ ] H1.0 `docs/PREDICTIONS-H1.md` committed before any run.
- [ ] H1.1 Integrators and Lyapunov spectrum (Benettin QR), float64.
- [ ] H1.2 Classifier and the verification checks.
- [ ] H1.3 Positive and negative controls (list in `PLAN.md`).
- [ ] H1.4 Cross-check against `lyapax` and, if easy, `jitcode`; pin versions.
- [ ] H1.5 Planted-bug check.

## H2 to H6
See `PLAN.md`. Tasks are written when the previous stage is approved.

## Open questions for the owner
- Make this a private GitHub repo so a cloud session can run it? (Not done; the folder is local only.)
- Is a 3D flow rediscovery (H2) worth doing before 4D, or go straight to 4D? The plan assumes yes: it calibrates the search.
