# H2E predictions and gates: pooling the survivors for the float32 pass, and a GPU benchmark (written 2026-10-27, before any code or run)

## Part 1: pooling (CPU)

The unpooled float32 pipeline (H2D) gave each worker only its own chunk's survivors (about 20) in a padded batch of 32. Pooled pipeline, three phases on seeds 1000-1059 (1.2e6 draws, same as H2D):
1. 6 workers: screen + boundedness filter on each 20000-draw chunk; the survivors' coefficient vectors are collected.
2. All survivors pooled and split into 6 equal slices; each of 6 processes runs the float32 pass (`f32_keep`, same rule, thresholds and step as H2D) on its slice in batches of 128.
3. The kept systems go to the float64 stage 2 and confirmation as before, 6 workers.

Nothing about the rule, thresholds or stage 2 changes, so recall is by construction the H2D recall. Only the batching changes.

**Gates**
- **GA (same hits).** The pooled run finds exactly the 60 hits of the unpooled H2D run on the same seeds: none lost, none new. Also the number of systems kept by the float32 pass is within 3 of the unpooled 161.
- **GB (speed).** Wall time of the pooled run against the unpooled H2D run's 464 s for the same seeds (that run had 6 workers; I will say what else was running). Pass: at most 80% of 464 s (1.25x or better). If the result is within 15% of the gate I rerun the unpooled pipeline right after for a same-conditions comparison.
- **GC (cost per system).** Float32 phase wall time times 6 divided by the number of pooled survivors, reported against the 0.25 s of the single-process timing at batch 32.

**Predictions:** P-E1 GA passes: 85%. P-E2 GB passes (1.25x or better): 65%; 1.5x or better: 30%. P-E3 float32 cost per system (6 processes, batch 128) is at most 0.3 s: 60%. P-E4 at least one prediction here is missed: 80%.

## Part 2: GPU benchmark (only a benchmark, no search run)

Owner asked for it on 2026-10-27. Rules from the 2026-10-26 incident apply: memory cap, watchdog, first run tiny and in the foreground under 2 minutes, an out-of-memory error or a slow tiny run is a stop sign, no background GPU job.

What is measured: seconds per system for the float32 spectrum (same algorithm, dt 0.01, T 5000) as a function of batch size, on the GPU, against the CPU numbers above. The path depends on what runs: JAX with CUDA needs WSL (Linux only), PyTorch with CUDA runs natively on Windows. If WSL is unreachable I use PyTorch with a hand-written 3x3 Gram-Schmidt in place of QR, and I say clearly that it is a different implementation of the same method, checked against the JAX float32 exponents on a few systems before any timing is believed.

**Predictions:** P-E5 the GPU float32 pass is at least 5x cheaper per system than the CPU float32 pass at batch 128 (0.21 s) at its best batch size: 60%; at least 20x: 30%. P-E6 the best batch size is 1024 or more: 70%. P-E7 the benchmark runs without exceeding 3 GB of GPU memory above the display's own use: 85%.

## Results (2026-10-27, written after the runs; no gate or prediction above was changed)

Files: `results/h2_h2e_pooled.json` (pooled run), `results/h2_h2e_unpooled_rerun.jsonl/.log` (unpooled H2D pipeline rerun right after), `results/h2e_gpu_bench_T*.json`, `results/h2e_cases.npy`, `results/h2e_gpu_check.npy`, scripts `experiments/h2_pooled.py` and `experiments/h2e_gpu_bench.py`. The machine is shared; I did not measure the background load, so only same-conditions comparisons count.

### Part 1: pooling (CPU)

| Gate | Result |
|---|---|
| GA same hits | **pass**: 60 hits, identical to the unpooled run (none lost, none new); float32 pass kept 161, the same as unpooled |
| GB speed (80% of 464 s or less) | **fail**: pooled wall 387 s, which is 83% of the earlier 464 s. The pre-registered rerun of the unpooled pipeline right after took 431 s, so the same-conditions gain is 1.11x |
| GC float32 cost per system | **pass, barely**: 65.4 s for 1366 systems on 6 processes, 0.287 s per system (includes each process compiling once) against the 0.25 s of the batch-32 single-process timing |

**Where the time goes after pooling** (60 seeds, 1.2e6 draws, 6 workers): phase 1 (screen + boundedness, still float64) 142 s; phase 2 (float32 pass) 65 s; phase 3 (float64 stage 2 and confirmation of the 161 kept systems, 60 of which are real hits) 180 s. Pooling only touches phase 2, so it cannot give more than the 65 s it costs. The float64 confirmation of real hits, 20000 time units from two starts, about 9 s each, is now the largest piece, and it cannot be skipped without changing the registered rule.

### Part 2: GPU benchmark (RTX 5070, PyTorch 2.11 with CUDA, no search run)

WSL (Ubuntu) did not answer (`HCS_E_CONNECTION_TIMEOUT` twice), so JAX-on-GPU was not possible. I did not restart WSL, because other agents may be using it. I used PyTorch on Windows instead (the existing `ai-detector-test` environment, read only): the same method written as batched tensor operations with a CUDA graph, Gram-Schmidt in place of QR. Memory was capped at 25% of the card with a watchdog at 8.5 GB; GPU memory in use stayed at the display's baseline plus under 0.1 GB, and every run was in the foreground and under 2 minutes.

**Agreement with float64 (T = 1000, 64 systems):** B 0.2076, -0.0014, -1.2063; S 0.1782, -0.0008, -1.1774; F 0.1234, 0.0021, -0.6255; O 0.0488, -0.0022, -0.3172. Float64 values at T = 5000: B 0.215, S 0.187, F 0.118, O 0.048. Differences are under 0.01 (short-run noise); the divergence is exact. At T = 200 the GPU and JAX-CPU float32 exponents of 8 hits differ by up to 0.43 because chaotic orbits diverge after rounding differs; the first system agrees to 4 digits.

**Cost per system for T = 5000 plus the 200-unit spin-up** (from runs at T = 200 and 600 for the two large batches, T = 200 only for the small ones, so the small-batch figures are rougher; extrapolated linearly):

| Batch | Seconds per system (GPU) |
|---|---|
| 256 | about 0.3 |
| 1024 | about 0.1 |
| 4096 | 0.028 |
| 16384 | 0.019 |
| CPU JAX float32, one process, batch 128 | 0.21 |
| CPU JAX float32, six processes in the pooled run | 0.048 aggregate (65.4 s for 1366) |
| CPU JAX float64, one at a time | 1.10 |

So the GPU is about 11 times cheaper than one CPU process, but only **about 2.5 times cheaper than all six CPU processes together**. At 16384 it was still limited by GPU work, not launch latency (T = 600 took 47 s against 23 s for T = 200, so 0.060 s per 100 steps for the whole batch). The operations are many small separate kernels, each reading and writing memory; the card runs far below its 30 TFLOPS. A single fused kernel in which each thread integrates one system in registers should be much faster. That is an estimate, not a measurement.

### What this means for the 1e9-draw run

Scaling the 60-seed phase times (142, 65, 180 s) by 833 gives about 33 h + 15 h + 42 h = 90 h, which is 3.8 days on 6 workers. A GPU float32 pass would take the 15 h phase off the CPU (the GPU would need about 7 h for it). It does nothing for the 42 h of float64 stage 2 and confirmation, nor for the 33 h of screening unless the screening is moved to float32, which has not been tested for recall. **So my earlier statement that a GPU would make this an overnight job was too optimistic.** The float64 part alone is at least a day. A 1e8-draw run is about 9 h with the current CPU pipeline, which is overnight.

**Predictions judged.** P-E1 true. P-E2 **missed** (1.20x against the earlier run, 1.11x same-conditions). P-E3 true (0.287 s; includes compile). P-E4 true (some predictions missed). P-E5 true against one CPU process (11x against the 5x asked), but against six CPU processes it is 2.5x. P-E6 true (best batch was the largest tested, 16384). P-E7 true.
