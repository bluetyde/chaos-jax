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
