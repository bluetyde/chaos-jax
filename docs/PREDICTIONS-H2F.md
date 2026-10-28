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
