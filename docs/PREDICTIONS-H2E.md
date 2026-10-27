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
