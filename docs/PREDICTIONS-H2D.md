# H2D predictions and gates: a float32 coarse spectrum screen before stage 2 (written 2026-10-27, after a calibration look, before the filter was wired in or any gate run)

## Why and what was looked at first

After the boundedness filter (H2C), about 42% of the stage-1b survivors still reach the float64 stage 2, which costs about 1.4 s per system run one at a time. Question from the owner: can a float32 pass act as a coarse finder, with every hit still confirmed in float64? A float32 Lyapunov spectrum can be run for a whole batch of systems at once (`vmap`), so even on the CPU it may be cheaper per system. This stage tests that on the CPU only. No GPU is used.

**Calibration data (disclosed; not used for any gate).** `experiments/h2d_calib.py` ran the float32 batched spectrum (dt 0.01, T 5000, spin 200, QR every 0.1) on the boundedness-filter survivors of seeds 2000-2005 (139 systems), next to their float64 stage-2 labels (`results/h2d_calib.json`, labels from `results/h2c_explore.json`). Float32 and float64 agree on cases B, S, F, O to about 0.005 in each exponent. Of the 139: 81 inconsistent, 23+9 quasi-periodic, 13 fixed point, 5 diverges, 8 chaotic. The 8 chaotic ones have float32 lambda_1 between 0.028 and 0.40, a second exponent within 0.001 of zero, and |sum of exponents - mean divergence| below 0.0025. The rule below was chosen after seeing this, so P-D3 is not independent of it; the gates run on other data.

## Definition fixed now

For each system that passed the boundedness filter, a float32 spectrum is computed (same method and step as stage 2, batch of 32). A system is **dropped** if any of:
1. any exponent, the mean divergence, or the largest |x| is not finite, or largest |x| is above 1e8;
2. |sum of exponents - mean divergence| > 0.05 * max(1, |mean divergence|) (the float64 stage-2 rule uses 1e-3; the float32 pass is lenient);
3. lambda_1 <= 0.005;
4. every exponent has |lambda_i| > 0.05 (no zero exponent, as a flow on an attractor must have).
Otherwise it goes to the float64 stage 2 as before. Stage 2 and confirmation are unchanged: a system can only become a hit through float64.

## Gates

- **GA (recall).** All 472 confirmed hits of `results/h2_main.jsonl` through the float32 filter alone (not through the boundedness filter). Pass: at least 99% (467) kept. Every dropped hit listed.
- **GB (same draws, same hits).** Pipeline screen -> boundedness -> float32 -> stage 2 on seeds 1000-1059 (1.2e6 draws). Pass: at least 95% of the old 63 hits recovered and none the old run did not find.
- **GC (speed).** On seeds 1000-1005, the new pipeline against the boundedness-only pipeline rerun now, back to back, same 6 workers. Pass: at most two thirds of the worker-seconds (1.5x or better). I will say what else was running.
- **GD (planted bugs).** Tests fail when the filter is broken in four ways: result inverted; sum-check tolerance made huge; lambda_1 threshold made hugely negative; zero-exponent limit made huge. Each needs a test vector that only that rule drops.
- **GE (report).** Survivors per stage, hits lost, speedup, float32 cost per chunk.

## Predictions (probability that the statement is true)

- P-D1: GA passes (467 or more of 472): 85%. All 472 kept: 55%.
- P-D2: the hit set on seeds 1000-1059 is identical to the old one: 70%.
- P-D3: at most 15% of the boundedness survivors reach stage 2: 75%. At most 10%: 50%. (Calibration gave 9.4%.)
- P-D4: GC passes (1.5x or better over boundedness-only): 70%. 2x or better: 40%.
- P-D5: the float32 pass costs less than the float64 stage 2 it replaces on the same survivors: 80%.
- P-D6: GD passes on the first run: 70%.
- P-D7: at least one prediction here is missed: 85%.

## What would change my mind

If GA fails, I look at which rule dropped each lost hit; a loose sum tolerance would be the first thing to change, with new data. If recall is full but the speedup is under 1.5x, the float32 pass on a CPU is not worth its cost and the only place float32 could still help is a GPU, which I would ask about first. A CPU float32 result says nothing about GPU speed.

## Results (2026-10-27, written after the runs; no gate or prediction above was changed)

Files: `results/h2d_recall.json`, `results/h2_h2d_new.jsonl/.log` (pipeline with the float32 screen, `H2_FILTER=f32`, seeds 1000-1059), `results/h2_h2d_bound6.jsonl/.log` (boundedness-only control rerun on seeds 1000-1005 right after), `results/h2d_timing.json`, `results/h2d_calib.json` (calibration, seeds 2000-2005). The machine is shared with other agents; I did not measure the load, so only same-conditions comparisons count. All of this is on the CPU. No GPU was used.

**The float32 screen is about 5 times cheaper per system than the float64 stage 2 but loses 2% of the known hits, and the pipeline gets only 1.28x faster. Two of the three gates fail.**

| Gate | Result |
|---|---|
| GA recall on the 472 known hits | **fail**: 463 of 472 kept (98.1%), 467 needed. Nine lost (see below) |
| GB same draws, same hits | **pass, barely**: 63 old hits, 60 recovered (95.2%, 95% needed), none new. The three missing (hits 31, 36, 51 of the main file, seeds 1027, 1034, 1046) are three of the nine lost in GA |
| GC speed (1.5x or better over boundedness-only) | **fail**: seeds 1000-1005 back to back, new 231 worker-seconds against 296 for the boundedness-only pipeline, 1.28x. (The earlier H2C run of the same seeds took 302.) |
| GD planted bugs | **pass**: bugs 9-12 all caught (12 of 12 in the script, tree clean after) |

**Why hits are lost.** Float32 rounding sends a chaotic orbit onto a different path than the float64 one. For most systems that does not matter, since the exponents are statistical. For these nine the float32 orbit ended up somewhere else: three overflowed to non-finite values (mean divergence 1896, 32567 and 38569, systems with very strong contraction), two have an exponent sum far from the divergence (an inconsistent run), and four have lambda_1 of 0.06 to 4 with no exponent near zero (a different, unstable regime). Hit 319 had exponents 0.136, -0.054, -1.183 in float32 against 0.081, -0.0007, -1.181 in float64. None of these is a rounding error of 1e-7 in the exponents; they are changes of behaviour. A looser rule would let more of them through and also more of the wrong ones. I did not tune it after seeing this.

**Survivors.** Of 1366 boundedness survivors, 161 reached stage 2 (11.8%); stage 2 found 70 chaotic and 60 confirmed (old and boundedness-only: 76 and 63).

**Cost.** One process, same 128 hits (`results/h2d_timing.json`): float32 batched 0.29 s per system at batch 16, 0.25 at 32, 0.21 at 128; float64 one at a time 1.10 s. So a full batch is 4.4 to 5.2 times cheaper per system. In the pipeline the batch of 32 is only about 60% full (about 20 survivors per chunk), and six processes compete for the cores, which is why the whole pipeline gains only 1.28x.

**Predictions judged.** P-D1 **missed** (463, not 467; not all 472). P-D2 **missed** (60 of 63). P-D3: at most 15% true (11.8%), **at most 10% missed**. P-D4 **missed** (1.28x). P-D5 true. P-D6 true. P-D7 true (five misses).

**What this says.** A float32 coarse finder works as an idea: it is cheap and it does drop 88% of the survivors. But on this problem it is not free, because float32 orbits of chaotic systems with strong contraction or sensitive attractors change behaviour, and the loss falls on real hits. Three options, none done: (1) accept a 2 to 5% loss and say so in every count; (2) use float32 only to rank, with float64 on everything above a lower bar, which costs speed; (3) keep survivors from many chunks in one pool so the batch is always full (cost per system 0.21 s rather than about 0.4), which would help the speed but not the recall. Nothing here speaks for a GPU: the GPU only changes the speed of this float32 pass, not which hits it loses.
