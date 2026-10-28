# PLAN-H2F: making the H2 search (and its successors) fast enough for a large run (written 2026-10-27, plan only, nothing run)

## Goal and where we start

Goal: a 1e9-draw run, or the same number of distinct hits some other way, in about one night (10-12 h), without losing any of the registered recall. Today (main f81171a, 6 workers, 1.2e6 draws in 387 s, the pooled run):

| Phase | Seconds | Share | At 1e9 draws |
|---|---|---|---|
| 1 screening (float64 stage 1a, 1b) + boundedness | 142 | 37% | about 33 h |
| 2 float32 spectrum pass | 65 | 17% | about 15 h |
| 3 float64 stage 2 + confirmation (60 real hits) | 180 | 47% | about 42 h |

A 1e8-draw run is already about 9 h. Every item below is gated like H2B-H2E: predictions and gates written first, calibration disclosed, recall measured on the 472 known hits and on the unchanged hit set of seeds 1000-1059 (60 hits), planted bugs for new code, same-conditions timing, results written without changing gates.

## Correction to what I said earlier (item 1)

I said removing symmetric duplicates could cut the draws by 10-48x. **That was wrong for a random search.** The family has 48 symmetries (6 renamings of x, y, z times 8 sign flips), so one system has up to 48 coefficient vectors. But a random draw hits a given orbit with the same chance whether or not duplicates are removed: the hit rate per draw (4.7e-5) is the same, and the space is about 6e17 vectors, so a 1e9-draw run almost never draws the same orbit twice. Restricting to one representative per orbit saves nothing. Item 1 is replaced by W1 below.

## Workstreams (your numbering: 1, 2, 3, 4, 6)

### W1. Fewer draws per hit (replaces the symmetry idea)
- **W1a, orbit deduplication of hits (cheap, no search change).** Reduce each hit to the lexicographically smallest of its 48 images, so the catalogue counts systems, not copies. Test: planted copies of known hits under random group elements must collapse; a non-equivalent pair must stay apart. Report how many of the 472 hits are orbit duplicates (I expect almost none).
- **W1b, adaptive support weighting.** The hit rate per draw varies strongly with the support (k = 6 gives 348 of 472 hits from half the draws; calibration shows published supports differ by a factor of several). Draw the support from a distribution weighted by hits seen so far (with a floor, so no support gets probability zero), then draw coefficients uniformly on it as now. Gate: hits per worker-second at least 2x the uniform sampler on fresh seeds, and every hit still confirmed in float64. The cost: counts of hits are no longer an estimate of abundance; I would label such a run "targeted" in every table. Needs your decision D3.

### W2. Fused CUDA screening (and later the float32 spectrum pass)
Order matters; the first step can kill the idea cheaply.
1. **W2a, does float32 screening keep the hits? (CPU, JAX).** Re-implement stage 1a and 1b in float32 in JAX on the CPU. Gates: G3 recall (the 245 screen-recall positives from H2) and all 472 hits through float32 stage 1. If more than 1% of hits are lost, GPU float32 screening is dropped here, before any kernel is written. This separates the numerical question from the kernel question.
2. **W2b, toolchain.** Findings today: no `nvcc`, no MSVC `cl`, no cupy, numba or triton on the machine; PyTorch 2.11 has the CUDA 12.8 runtime. WSL Ubuntu hung (`HCS_E_CONNECTION_TIMEOUT`). Choices: (i) a new venv with `cupy-cuda12x`, which compiles through NVRTC from pip wheels and needs no MSVC (about 100 MB download; needs your OK, D2); (ii) no install, using PyTorch's NVRTC through `torch.cuda.jiterator`, which only allows elementwise kernels with few inputs and is awkward for 30 coefficients. My recommendation is (i).
3. **W2c, kernel.** One thread per system, state in registers, the flow and the tangent step unrolled; same arithmetic and order as the W2a JAX code so decisions can be compared draw by draw. Gate: the keep/drop decision agrees with the JAX float32 code on at least 99.9% of 1e5 draws, and every disagreement is listed.
4. **W2d, speed.** Seconds per 1e6 draws against the CPU's 142 s per 1.2e6 (phase 1). Prediction to be written before the run: at least 20x. Same GPU protocol as before: memory cap, watchdog, tiny foreground first run, nothing in the background, nothing longer than a few minutes unattended without your word.
5. **W2e, integration.** The GPU writes survivors to disk in blocks (pooled, as in H2E); the CPU float64 stage 2 and confirmation consume them at the same time, so the card and the 8 cores both work. Optional afterwards: the float32 spectrum pass as a second kernel (phase 2, 15 h of CPU moved to the GPU).

### W3. Early exit in screening
- **W3a, measure first (CPU).** For stage-1a and 1b survivors and failures of seeds 2000-2005 (calibration), record the time at which each system dies (escape, fixed point, or the point where lambda_1 is clearly settled). Output: a histogram and the saving a perfect early exit would give.
- **W3b, rounds with compaction (JAX, CPU).** Run stage 1b in rounds of 100-200 time units and drop the systems that are already decided; the batch shrinks between rounds. Gate: identical keep set on seeds 1000-1059 (3258 stage-1b survivors) and a measured speedup. On the GPU kernel the same exit is a `break` per thread; W3 informs whether it is worth coding there.

### W4. Cheaper screening settings (last, lowest expected gain)
Each is a separate variant with its own recall gate, run only after W3 data: larger step (dt 0.02 to 0.04, checked against dt 0.01 on the 472 hits), shorter horizon of stage 1b (600 to 300) with a sequential stop, fewer stage-1a steps. Any variant that loses a known hit is rejected, not tuned.

### W6. Phase 3 (float64 stage 2 and confirmation), the largest piece
- **W6a, 8 workers.** The CPU has 8 cores. Measure 6 against 8 processes on the same work (not 16: those are hyperthreads). Expected gain about 1.2-1.3x; the machine is shared, so this is for nights.
- **W6b, batch the two confirmation starts (JAX vmap).** Confirmation runs T = 20000 from two starts one after the other. Putting both in one batched call should cost less than twice. Gate: identical confirm decisions on all 472 hits and exponents within 1e-6 of the sequential values.
- **W6c, lazy confirmation per class.** Confirm in float64 only one representative per class (or per orbit) first, confirm the rest later or never. **This changes the registered rule ("every hit confirmed at T = 20000 from two starts"), so it needs your decision D1.** It is the only item that gives a large cut in phase 3: 472 hits fall in 266 classes at 1e7 draws, and I expect the ratio of hits to classes to grow with the run size.
- **W6d, float64 on the GPU (optional, low priority).** Consumer cards run float64 at 1/64 of float32; the card's float64 rate is about the CPU's, so a fused float64 kernel for stage 2 might add 1x on top of the CPU, not 20x.

## Order and dependencies

1. W3a (measure), W1a (orbit dedup), W6a (8 vs 6 workers): all CPU, independent, small. About half a day of my work.
2. W2a (float32 screening recall). Decision point: if it fails, W2c-e are cancelled and the plan reduces to W3, W4, W6, W1b.
3. W6b, W3b, then W1b if you approve D3.
4. W2b-d after your answers to D2 (install) and the go-ahead for the first GPU run.
5. W2e integration run; then W4 if screening is still the bottleneck; then W6c if you approve D1.
6. Final: a 1e8-draw run with everything on, compared to the 9 h baseline, then the 1e9 projection.

## Projection (estimates, not measurements)

| Setup | 1e9 draws |
|---|---|
| Today, pooled CPU, 6 workers | about 90 h |
| W2 GPU screening + float32 pass on the card, CPU does phase 3 | phase 3 only, about 42 h |
| plus W6a, W6b (8 workers, batched confirmation) | about 25-30 h |
| plus W6c (lazy confirmation, if D1) | about 8-12 h, i.e. overnight |
| plus W1b (targeted sampling, if D3) | the same hits in fewer draws, hence fewer hours again |

Without D1 or W1b a 1e9 run is not overnight, because phase 3 stays above a day. A 1e8 run would become about 1-3 h.

## Decisions I need from you
- **D1.** Allow lazy confirmation per class (changes the registered "confirm every hit" rule)? My recommendation: yes for the large run, with the full set confirmed afterwards in the background and the lazy and full counts both reported.
- **D2.** Install `cupy-cuda12x` in a new venv for the fused kernel (about 100 MB, no MSVC needed)? Recommendation: yes, after W2a passes.
- **D3.** Allow a "targeted" sampler (W1b) for runs whose goal is a catalogue, labelled as targeted and not used for abundance estimates?
- **D4.** Use 8 workers at night (the other agents share the machine)?

## Risks
- Float32 screening may lose hits the way the float32 spectrum pass did. That is why W2a comes first.
- The GPU also drives your displays: the 2026-10-26 incident rules apply to every GPU run (memory cap, watchdog, foreground tiny run first, stop on out-of-memory or on a slow tiny run).
- Speed gains of different items do not multiply cleanly; I will measure the combined pipeline, not add up the pieces.
- Targeted sampling and lazy confirmation both change what a "count" means; the tables must say which run produced them.

## Decisions and amendments (2026-10-27, later the same day)

- **D2:** wait until W2a has passed, then install `cupy-cuda12x` in a new venv (owner: "wait until W2a, then yes").
- **D4:** 8 workers are allowed at night; the owner will not be using the computer and the other agents are not running heavy jobs then. W6a still measures the gain first.
- **D1 (amendment A1, dated 2026-10-27, to take effect only for the large run, before it is started):** confirmation may be done in two tiers. Tier 1: one hit per support class (the first found in the run) is confirmed at T = 20000 from two starts, as now. Tier 2: all other hits stay in the list marked "stage 2 passed, not yet confirmed" and are confirmed afterwards or not at all. Every table of the run reports three numbers: found, confirmed, not yet confirmed. Facts that go with this amendment: stage 2 hits fail confirmation in 14% (60 of 70, 1.2e6-draw run) to 23% (472 of 610, main run) of the cases, so unconfirmed candidates are not hits and must never be counted as such; a confirmed class member says little about its classmates (a class is a pattern of terms, not a set of coefficient values); and the saving is probably 2-5x on phase 3, not the 5-10x I first guessed (472 hits fell in 266 classes at 1e7 draws, a ratio of 1.8). The amendment does not lower recall: no candidate is dropped, only the confirmation is postponed. It is implemented as W6c, after W6a and W6b, with its own gates.
- **D3:** the frozen-weights version is to be tested (owner). The adaptive version is not. See W1b results in `docs/PREDICTIONS-H2F.md`.

## What makes a speedup safe or unsafe (working rule)
1. **Lossless**: same inputs give the same decisions, only the computation is organised differently (batching, pooling, more workers, early exit that cannot change the decision). Safe if a test on the known hits and on seeds 1000-1059 shows identical decisions.
2. **Lossy and measured**: it can drop candidates (the boundedness filter lost 2 of 472 known hits; the float32 spectrum pass lost 9). Acceptable only if the loss is measured on known hits, stated in every count that depends on it, and small. A blind spot remains: every recall test uses hits that the old pipeline found, so true hits that both pipelines miss are invisible.
3. **Changes what a number means**: sampling weights, confirmation tiers. Allowed with a dated amendment and labels on every table; never mixed with uniform, fully confirmed data.
