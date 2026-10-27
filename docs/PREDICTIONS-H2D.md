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
