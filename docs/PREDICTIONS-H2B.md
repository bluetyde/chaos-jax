# H2B predictions and gates: a middle filter before stage 2 (written 2026-10-27, before any code or run)

## What this asks

H2 spent most of its time in stage 2 (full spectrum, T = 5000) on about 27,000 stage-1 survivors per 1e7 draws, of which only 610 passed. Question: can a cheaper, lenient test placed between stage 1 and stage 2 drop most of the failing survivors without dropping real hits? "Stopping failing runs early" is the same idea: a survivor that fails at T = 300 or T = 1000 never gets the 5000 run.

Old reference, from `results/h2_main.jsonl` (registered grid, 6 workers, seeds 1000-1499): 472 confirmed hits in 1e7 draws. For seeds 1000-1059 (1.2e6 draws) the old run took 502 worker-seconds per 1e5 draws (sum of the per-chunk times; the machine was shared with other agents, so this is not a clean timing).

## Definition fixed now

Middle filter: a ladder of two short spectrum runs of the H1 evaluator (`spectrum_p`, dt = 0.01, same start), tier 1 at T = 300 with spin 100, tier 2 at T = 1000 with spin 200. A system is **dropped** at a tier if any of:
- the run diverges (not finite or max |x| above 1e8) or the orbit is a fixed point (the H1 classifier says so);
- lambda_1 + 3 * SE_1 <= 0.005 (the largest exponent is not clearly above zero even with a three-sigma allowance).
Otherwise it goes to the next tier, and after tier 2 to stage 2 as before. Stage 2 and confirmation are unchanged. Deliberately lenient: "inconsistent", "conservative" and "hyperchaotic" labels are not used to drop anything at the middle tiers, because short runs make those labels noisy.

## Gates

- **GA (recall on known hits).** All 472 confirmed hits of `results/h2_main.jsonl` are put through the middle filter alone. Pass: at least 95% (449 of 472) are kept. Every dropped hit is listed with its class and its exponents.
- **GB (same draws, same hits).** The new pipeline is run on seeds 1000-1059 (the same 1.2e6 draws as the old run). Pass: it recovers at least 95% of the old confirmed hits for those seeds, and finds no confirmed hit the old run did not (the filter only removes; a new hit would mean the code changed something else).
- **GC (speed).** Worker-seconds per 1e5 draws on those seeds drop to half of the old 502 or less. Not a clean timing unless the machine is idle; I will check and say what else was running.
- **GD (planted bug).** The tests for the filter fail when the filter is broken in two ways: the drop test inverted, and the threshold made so large that nothing is ever dropped (the latter must fail a test that checks that sinks and divergent systems are dropped).
- **GE (report).** Survivors per tier, share of stage-1b survivors reaching stage 2, hits lost, speedup, written to `docs/PREDICTIONS-H2B.md` (Results).

## Predictions (probability that the statement is true)

- P-B1: GA passes (95% or more of the 472 kept): 90%. All 472 kept: 60%.
- P-B2: after both tiers, at most 15% of the stage-1b survivors reach stage 2: 70%. At most 30%: 90%.
- P-B3: GC passes (2x or better): 75%. 3x or better: 45%.
- P-B4: the hit set on seeds 1000-1059 is identical to the old one: 55%.
- P-B5: the hits the filter loses, if any, have a small lambda_1 (below 0.05): 80%.
- P-B6: GD passes on the first run: 80%.
- P-B7: at least one prediction here is missed: 90%.

## What would change my mind

If GA fails (more than 5% of real hits dropped), the filter is not safe as a screen for the registered search and the main run stays as is; the loss pattern (small lambda_1 only?) tells me whether a longer tier 2 would fix it. A speedup under 2x with full recall means the cost is in stage 1 or in confirmation, and the next step is to profile before adding anything.

## Results (2026-10-27, written after the runs; no gate or prediction above was changed)

Files: `results/h2b_recall.json`, `results/h2_h2b_new.jsonl/.log` (new pipeline, seeds 1000-1059, 1.2e6 draws, `H2_MID=1`), `results/h2_h2b_old6.jsonl/.log` (old pipeline rerun on seeds 1000-1005 for a same-conditions timing), `experiments/h2b_profile.py`. The machine was shared with other jobs (about 28% CPU busy before I started), so timings are comparisons within the same conditions, not absolute numbers.

**The middle filter is safe but does not speed anything up. It made the pipeline slightly slower.**

| Gate | Result |
|---|---|
| GA recall on the 472 known hits | **pass**: 472 of 472 kept |
| GB same draws, same hits | **pass**: 63 old hits, 63 new, all identical, none new, none lost |
| GC speed (2x or better) | **fail**: seeds 1000-1005 run back to back, new 573 worker-seconds against 534 for the old pipeline, speedup 0.93 (the old run's file gave 502 and the new run 516 per 1e5 draws over 60 seeds) |
| GD planted bugs | **pass**: both new bugs caught (6 of 6 in the script) |

**Why it fails.** The filter drops only 25% of the stage-1b survivors (2453 of 3258 reach stage 2; stage 2 still sees 76 chaotic ones, as before). Profile of one chunk (seed 1000, one process): screening 10 s; middle filter on all 54 survivors 13.5 s; stage 2 on all 54 survivors 55 s. The middle filter costs about a third of a stage-2 run per system and saves a stage-2 run for only one system in four, so it costs more than it saves. The reason is in how the survivors fail at stage 2: of the 54, 13 diverge, 23 are labelled "inconsistent", 7 are fixed points, 8 are conservative quasi-periodic, 3 are quasi-periodic. Divergence and "inconsistent" (the sum check or the zero exponent fails, i.e. unbounded or drifting orbits) are late events that a T = 300 or T = 1000 run does not see, and most survivors already have a clearly positive largest exponent at short times because stage 1b selected them for that.

**Predictions judged.** P-B1 true (all 472 kept). P-B2 **missed** (75% reach stage 2; I said at most 15%, and at most 30% at 90%). P-B3 **missed** (0.93x; I said 75% for 2x). P-B4 true (identical hit sets). P-B5 not tested (no hit lost). P-B6 true. P-B7 true (two misses).

**What this says about the next step.** "Stop failing runs early" needs a test that sees what actually kills the survivors: orbits that escape or drift late, not a short exponent. 43 of 54 survivors fail by escape or drift. A state-only run over a long horizon (no tangent frame, no QR; much cheaper per step, and batchable like stage 1) with a boundedness test would catch them. I have not tried it and have not measured how much it would save; a written design and predictions come first. The middle filter stays in the code, off by default (`H2_MID=1` turns it on), because it is correct and tested, just not useful.
