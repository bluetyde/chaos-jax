# H2F predictions and gates: orbit deduplication, frozen sampling weights, screening profile, 8 workers, float32 screening recall (written 2026-10-27, before any code or run)

Five CPU-only steps from `docs/PLAN-H2F.md` (W1a, W1b, W3a, W6a, W2a). No GPU is used. Nothing here changes the registered H2 rules; every number below is about a measurement or an offline analysis, and the sampler of W1b is only evaluated, not used for a catalogue.

## W1a. Orbit group and deduplication of the 472 hits

The family has 48 symmetries: the 6 renamings of (x, y, z) times the 8 sign flips of the variables. For a renaming `pi` and signs `s`, the new variables are `y_i = s_i x_pi(i)` and the new system is `g(y) = S P f(P^T S y)`, which stays on the coefficient grid. The canonical form of a coefficient vector is the lexicographically smallest of its 48 images.

**Gates**
- **GF1a.** For 5 random systems and all 48 group elements, the transformed system satisfies `g(y) = S P f(x(y))` at 10 random points to 1e-12 (the algebra is right, not just self-consistent).
- **GF1b.** For 20 random vectors, all 48 images have the same canonical form; a generic vector has 48 distinct images; changing one coefficient of a vector changes its canonical form.
- **GF1c.** Planted bugs (sign factor of a monomial dropped; renaming applied to equations only) are caught by the tests.
- **GF1d (report).** How many of the 472 hits are orbit duplicates of another hit.

**Predictions:** P-F1: GF1a-c pass on the first run: 70%. P-F2: at most 2 of the 472 hits are orbit duplicates: 90%; none at all: 70%.

## W1b. Frozen sampling weights, offline test

Model: for a support (the set of non-zero slots) with k terms, the hit rate is proportional to `w(S) = exp(sum of theta_slot over the slots in S)`, separately for k = 5 and k = 6. The `theta` are fitted (ridge-penalised, penalty chosen by 5-fold cross-validation on the training hits only) on the **training hits** (seeds 1000-1249 of `results/h2_main.jsonl`), each hit enlarged by its 48 symmetric images (the hit rate is invariant under the symmetries, so this is legitimate extra data). Weights are floored at 0.1 of the mean so no support is shut out. Test hits: seeds 1250-1499, not seen in fitting.

Predicted gain of a weighted sampler in hits per screened draw = (mean of `w` over the test hits) / (mean of `w` over uniformly drawn supports of that k). A gain of 1 means no better than uniform.

**Gates.** GF2 (offline): report the gain for k = 5 and k = 6 on the test hits, plus the gain of a trivial weighting that only changes the share of k = 5 and 6. Pass for "worth a real run": gain at least 1.3 for at least one k. If it passes, one real run on fresh seeds (3000 onward) with the weights follows, to measure hits per draw against the uniform sampler on the same seeds. If it fails, W1b stops there.

**Predictions:** P-F3: gain at least 1.3 for k = 6: 50%; for k = 5: 35%. P-F4: gain at least 2.0 for either k: 15%. P-F5: the trivial k-share weighting beats the slot model on the test hits: 35%.

A note on the aim. The project wants small systems. H2 draws k = 5 or 6 because Sprott's published 3D cases have 5 or 6 terms. Whether 4-term chaotic flows exist in this family was not tested by H2. A weighting that favours k = 6 would work against "simplest", so gains are reported per k, and weights are never used to shift the k mix toward more terms.

## W3a. Where stage 1b spends its time (calibration seeds 2000-2005)

Stage 1b runs 1200 chunks of 25 steps for every stage-1a survivor and decides at the end (largest-exponent estimate above 0.01, and bounded). Record per system the chunk of first excursion past 1e4 (or non-finite) and the running estimate at chunks 300, 400, ..., 1200. Then, offline, find the earliest checkpoint at which simple rules decide correctly: R1 drop on excursion; R2 drop at chunk t if the running estimate is below 0.

**Measurement only (no gate).** Output: how many systems are decided by which checkpoint, and the fraction of stage-1b steps a compaction scheme with those rules would save, with the number of final keepers it would wrongly drop (the target is none among the roughly 330 keepers of these seeds).

**Predictions:** P-F6: at least half of all stage-1b steps are spent on systems whose final decision a simple rule has determined by chunk 600: 70%. P-F7: a rule built from R1 and R2 saves at least 1.5x of stage-1b time without dropping a calibration keeper: 55%. These rules are chosen on seeds 2000-2005; any gate for them (W3b) runs on seeds 1000-1059.

## W6a. 6 against 8 worker processes

The same pooled pipeline (`experiments/h2_pooled.py`) on seeds 1000-1031 with 6 workers and then with 8, back to back, with no other heavy job running (I will look at the process list and CPU load before and after, and say what I found).

**Gate (report).** Wall time for both, per phase. **Predictions:** P-F8: the 8-worker run is at least 1.15x faster in total: 55%; at least 1.3x: 20%. P-F9: the two runs find the identical hits: 95%.

## W2a. Does float32 stage 1 keep the screening keepers? (CPU, JAX)

Stage 1a and 1b re-implemented with float32 arrays (same steps, same thresholds), run on seeds 1000-1059 next to the float64 stage 1 on the same draws.

**Gates**
- **GF3a.** At least 99% of the float64 stage-1 keepers (about 3258) are also keepers in float32. Every lost keeper is listed.
- **GF3b.** All 472 known hits pass float32 stage 1 (at least 99%, 467).
- **GF3c (report).** Float32 keepers that float64 drops (extra work for later stages), and the CPU time of both versions.

If GF3a or GF3b fails, the GPU kernel work (W2c-e) is cancelled.

**Predictions:** P-F10: GF3a passes: 55%. P-F11: GF3b passes: 75%; all 472 pass: 40%. P-F12: float32 keepers that float64 drops are at most 20% extra: 70%. P-F13: float32 stage 1 on the CPU is not faster than float64 (within 20%): 70%.

## Overall

P-F14: at least four of the predictions P-F1 to P-F13 are missed: 70%.

## Results (2026-10-27, written after the runs; no gate or prediction above was changed)

Files: `results/h2f_orbits.json`, `results/h2f_weights.json`, `results/h2f_weights_theta.npz`, `results/h2_h2f_weighted.json`, `results/h2f_profile1b.npz`, `results/h2f_profile1b_rules.json`, `results/h2_h2f_w6.json`, `results/h2_h2f_w8.json`, `results/h2f_screen32.json`, `results/h2f_breakit.txt`; scripts `experiments/h2f_*.py`, `src/hyperchaos/orbits.py`, `weighted.py`, `screen32.py`. The machine was not otherwise busy (background load about 5% of the cores, from desktop programs; I looked before the timing runs). Everything is CPU only.

### W1a. Orbits: pass, with no duplicates
GF1a (transformed flow equals the change of variables at random points, 5 systems x 48 elements) and GF1b (same canonical form for all 48 images; 48 distinct images for a generic vector; a changed vector has a different orbit) pass. GF1c: planted bugs 13 (sign factor dropped) and 14 (equations not renamed) are both caught. One test failed on its first run, and the fault was in my test, not in the code: I had used the published case B as a "generic" vector, but B has a symmetry of its own and only 24 distinct images. The test now uses a random full vector. **Among the 472 hits there are 472 distinct orbits: no orbit duplicates.** 383 hits have 48 images; 89 have 24 (they carry a symmetry of their own).

### W1b. Frozen weights: the offline test and a real run both show a large gain per draw
Offline (fitted on seeds 1000-1249 with each hit enlarged by its 48 images, ridge penalty by 5-fold cross-validation on training hits only, floor 0.1; tested on the 55 k = 5 and 156 k = 6 hits of seeds 1250-1499):

| | k = 5 | k = 6 |
|---|---|---|
| predicted gain, hits per draw | 4.5 | 3.0 |
| control (training labels shuffled across slots) | 0.60 | 1.09 |
| acceptance rate of the rejection sampler (cap at the 99.9th percentile) | 3.8% | 6.4% |

The k-share weighting alone (all draws on k = 6) would give 1.47, and it would push toward more terms.

Real run (`H2_WEIGHTS=results/h2f_weights_theta.npz`, seeds 3000-3059, 1.2e6 draws, 6 workers, same pooled pipeline, same float32 pass, confirmation unchanged): **187 confirmed hits, against 60 for the uniform sampler on seeds 1000-1059 (and 56.6 expected from the main run's rate): 3.1x to 3.3x per draw.** By term count: k = 5: 68 hits against about 16 expected (4.3x; offline prediction 4.5); k = 6: 119 against about 41 (2.9x; prediction 3.0). The k mix of draws was left at one half each. Diversity: the 187 hits fall in 101 support classes, 48 of which are in the main run's list of 266 and 53 not; none of the 187 coefficient vectors is in the same orbit as a main-run hit.

**But the time gain is small: 187 hits in 983 s against 60 in 387 s, which is 1.23x per second.** The weighted sampler sends 3.1x more systems through every stage (bound survivors 4192 against 1366; stage 2 216 against 70), and the confirmation of real hits is the bottleneck: phases 183, 210 and 590 s against 142, 65 and 180. A targeted sampler pays off in hits per draw, not in hits per hour, until confirmation gets cheaper (W6b, D1).

### W3a. Stage 1b early exit: only a small lossless gain
On 27,990 stage-1a survivors of seeds 2000-2005 (343 final keepers; my instrumented copy gives exactly the keepers of `search3d.screen` on every seed): 28.6% of the systems leave |x| = 1e4 at some point (4.8% by chunk 100, 15.6% by 200, 24.8% by 600). Stopping a system as soon as it escapes loses no keeper and saves **1.27x to 1.28x** of stage-1b time with an immediate exit (a GPU thread can do that; with rounds of compaction every 100 chunks 1.23x). Rules based on the running exponent gain more but drop keepers: threshold 0 at checkpoints every 100 chunks gives 1.70x and loses 44 of 343 keepers (13%), at 600 only 16 of 343 (5%); a threshold of -0.1 or lower loses none but adds nothing over the escape rule. The reason: the survivors that finally fail have running estimates close to those of the keepers (a true keeper has exponent 0.01 to 0.4), so the exponent cannot separate them early.

### W6a. 8 against 6 workers: 1.20x
Seeds 1000-1031, pooled pipeline: 183.6 s with 8 workers against 219.5 s with 6, same 33 hits. Phase 1 1.19x, phase 3 1.32x; phase 2 did not speed up (89 systems split over more processes still pay for a full batch of 128 each).

### W2a. Float32 stage 1: fails the recall gates, so the GPU screening kernel is cancelled
| Gate | Result |
|---|---|
| GF3a float64 keepers also kept in float32 (99% needed) | **fail**: 2658 of 3258 kept (81.6%), 600 lost. 40% of the lost have a float64 exponent below 0.05 (borderline), the rest are mostly large (median 0.22, 90th percentile 361: fast-growing systems) |
| GF3b known hits kept (467 of 472 needed) | **fail**: 464 of 472 (98.3%). Lost: indices 51, 133, 150, 159, 168, 309, 402, 426; three of them have exponent 0.013 to 0.027 |
| GF3c extra float32 keepers / CPU time | 612 extra (18.8% of 3258); float32 on the CPU is 2.2x slower than float64 (1710 s against 784 s over 60 seeds) |

Under the plan's rule, W2c-e (fused CUDA screening kernel) is cancelled, and D2 (installing cupy) is not triggered. A float32 screen loses about 2% of the real hits, the same as the float32 spectrum pass; using both would compound the loss. A variant with a looser threshold in float32 (to recover borderline systems at the price of more survivors) was not tested.

### Predictions judged
P-F1 **missed** (one test failed on its first run: my test, not the code). P-F2 true (none). P-F3 true (k = 6 gain 2.97, k = 5 gain 4.49). P-F4 true (both above 2.0; I had said 15%). P-F5 **missed** (the k-share weighting is far worse). P-F6 true (54.7%). P-F7 **missed** (the best rule that loses no keeper is 1.28x). P-F8 true (1.20x), 1.3x not reached. P-F9 true. P-F10 **missed**. P-F11 **missed** (464, and not all 472). P-F12 true (18.8%). P-F13 true (float32 is slower, 2.2x). P-F14 true (five misses).

### Planted bugs (all steps)
Bugs 1-14 (including 13 and 14 for the orbit transform): all caught, tree clean afterwards (`results/h2f_breakit.txt`, run before the weighted sampler existed). Bug 15 (weighted sampler acceptance inverted) was first run inside the full script and **hung**: with zero weights the inverted test never accepts, and the sampler looped forever, so the test never finished. I killed the run, restored the file from git, gave the sampler a loop guard that raises an error after 500 rounds, and ran bug 15 alone: caught (exit 1, tree clean). The full 15-bug script was not rerun after the guard.
