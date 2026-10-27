# TODO

Stage order is H0 to H6 (see `PLAN.md`). Do one stage at a time and wait for the owner's approval between stages.

## H0. Literature table (done on branch `claude/h0-literature`, awaiting owner review; two boxes left open)
- [ ] H0.1 Read Rössler 1979 (Phys. Lett. A 71, 155) in full. Copy the equations and the standard parameters. Fill the `LITERATURE.md` row; mark the "from memory" equations there as verified or wrong.
  Open: the 1979 paper itself is paywalled (HTTP 403), listed in `NEEDED-FROM-OWNER.md`. The equations and parameters are verified (correct) against Wilczak, Serrano and Barrio 2016, Eq. (2), and the 9-term count against Li and Sprott 2014; no exponent values found.
- [x] H0.2 Read Chlouverakis and Sprott 2006 (hyperjerk, Chaos Solitons Fractals 28, 739). Copy every n=4 and n=5 chaotic and hyperchaotic case with its equation and term count.
- [x] H0.3 Read Sprott 1994 and the Sprott and Linz 2000 review. Copy the list of 3D cases (A to S) with equations.
  Done with limits: Table I copied (checked by counting terms); Sprott 1994 pages 3-4 and most of Sprott and Linz were not read in full (`LITERATURE.md` section B).
- [ ] H0.4 Find and read the single-nonlinear-term 4D hyperchaos papers (Acta Phys. Sin. 58, 5201 and the 2010 J. Comput. Appl. paper). Get equations and term counts.
  Open: the 2009 Acta Phys. Sin. paper was read in part (system, 9 terms, 1 nonlinear, exponents); the 2010 J. Comput. Appl. paper is abstract only (PDF link dead), listed in `NEEDED-FROM-OWNER.md`.
- [x] H0.5 Read Kopp 2025 and Li and Sprott's 4D simplified Lorenz paper. Same rows.
  Done with limits: Li and Sprott read in full; Kopp 2025 and the related Kopp and Samuilik 2025 paper read in part.
- [x] H0.6 Check Sahle's 2005 thesis (Tübingen) and Wilczak, Serrano and Barrio for the Rössler hyperchaos proof. Note what each shows.
  Done with limits: Wilczak et al. 2016 read in part; Sahle thesis read in part (text layer lost digits); no 4D term count is printed there, my N = 4 reconstruction is marked as mine.
- [x] H0.7 Final table: smallest known 4D hyperchaotic flows by `terms` and by `nonlinear`, from the equations themselves. State the uncertainty (papers not found).
  Done: `LITERATURE.md` section A; best known is 7 terms, and 7 terms with 1 nonlinear term (Chlouverakis and Sprott Eq. 13).
- [x] H0.8 List the papers the owner must supply (paywalled) in `NEEDED-FROM-OWNER.md`.

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
