"""H2 long run in resumable blocks with per-phase time limits (for overnight use).
Usage: python -u experiments/h2_night.py <tag> <workers> <n_blocks> <seeds_per_block> <seed0> [batch]
Each block is 20000 * seeds_per_block draws, run as in h2_pooled.py (screen + boundedness, pooled float32 pass, float64 stage 2 + confirmation) and
written to results/h2_<tag>_b<NNN>.json when it finishes. Rerunning the same command skips finished blocks, so a stopped run resumes.
Environment: H2_WEIGHTS=<npz> uses the frozen-weight sampler (labelled targeted); H2_PHASE_LIMIT_S (default 5400) is the longest any one phase of a
block may take; H2_SKIP_F32=1 skips the float32 pass: if no result arrives in that time the worker processes are killed and the script exits with code 2 (finished blocks stay on disk)."""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutureTimeout

import numpy as np

import h2_pooled as hp

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
LIMIT = float(os.environ.get("H2_PHASE_LIMIT_S", "5400"))


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def run_phase(ex, fn, items, label):
    futs = [ex.submit(fn, x) for x in items]
    where = {f: i for i, f in enumerate(futs)}
    out = [None] * len(futs)
    try:
        for f in as_completed(futs, timeout=LIMIT):
            out[where[f]] = f.result()
    except FutureTimeout:
        log("STOP: %s produced no complete result within %.0f s; killing the worker processes" % (label, LIMIT))
        for p in list(ex._processes.values()):
            p.terminate()
        ex.shutdown(wait=False, cancel_futures=True)
        sys.exit(2)
    return out


def run_block(ex, tag, b, seeds, batch, workers):
    t_all = time.time()
    t0 = time.time()
    p1 = run_phase(ex, hp.w_phase1, seeds, "phase 1 (screening)")
    t1 = time.time() - t0
    pool = np.concatenate([p["cs"] for p in p1])
    t0 = time.time()
    if os.environ.get("H2_SKIP_F32"):  # no float32 pass: every boundedness survivor goes to the float64 stage 2 (slower, loses fewer hits)
        kept = pool
    else:
        keeps = run_phase(ex, hp.w_phase2, [(sl, batch) for sl in np.array_split(pool, workers)], "phase 2 (float32 pass)")
        kept = pool[np.concatenate(keeps)]
    t2 = time.time() - t0
    t0 = time.time()
    p3 = [r for r in run_phase(ex, hp.w_phase3, list(kept), "phase 3 (stage 2 and confirmation)") if r is not None]
    t3 = time.time() - t0
    hits = [r["hit"] for r in p3 if r["hit"] is not None]
    return dict(tag=tag, block=b, seeds=[seeds[0], seeds[-1]], draws=20000 * len(seeds), keep_a=sum(p["keep_a"] for p in p1), keep_b=sum(p["keep_b"] for p in p1),
                bound=int(len(pool)), f32=int(len(kept)), stage2=len(p3), confirmed=len(hits), t_phase1=t1, t_phase2=t2, t_phase3=t3, wall=time.time() - t_all,
                targeted=bool(os.environ.get("H2_WEIGHTS")), hits=hits)


if __name__ == "__main__":
    tag, workers, n_blocks, spb, seed0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    batch = int(sys.argv[6]) if len(sys.argv) > 6 else 128
    log("run %s: %d blocks of %d seeds (%.2e draws each), %d workers, seeds from %d, weights=%s, phase limit %.0f s" %
        (tag, n_blocks, spb, 20000 * spb, workers, seed0, os.environ.get("H2_WEIGHTS", "none"), LIMIT))
    t_start = time.time()
    done_hits = done_draws = new_blocks = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for b in range(n_blocks):
            path = os.path.join(RES, "h2_%s_b%03d.json" % (tag, b))
            if os.path.exists(path):
                r = json.load(open(path))
                log("block %d already on disk (%d hits), skipped" % (b, r["confirmed"]))
            else:
                seeds = list(range(seed0 + b * spb, seed0 + (b + 1) * spb))
                r = run_block(ex, tag, b, seeds, batch, workers)
                tmp = path + ".tmp"
                json.dump(r, open(tmp, "w"))
                os.replace(tmp, path)
                new_blocks += 1
                el = time.time() - t_start
                log("block %d done: %d hits, wall %.0f s (phases %.0f/%.0f/%.0f), elapsed %.1f h, estimated finish in %.1f h" %
                    (b, r["confirmed"], r["wall"], r["t_phase1"], r["t_phase2"], r["t_phase3"], el / 3600, el / new_blocks * (n_blocks - b - 1) / 3600))
            done_hits += r["confirmed"]
            done_draws += r["draws"]
    log("all %d blocks done: %d draws, %d confirmed hits" % (n_blocks, done_draws, done_hits))
    blocks = [json.load(open(os.path.join(RES, "h2_%s_b%03d.json" % (tag, b)))) for b in range(n_blocks)]
    summ = {k: sum(x[k] for x in blocks) for k in ("draws", "keep_a", "keep_b", "bound", "f32", "stage2", "confirmed", "t_phase1", "t_phase2", "t_phase3", "wall")}
    summ.update(tag=tag, targeted=blocks[0]["targeted"], blocks=n_blocks, seeds_per_block=spb, seed0=seed0)
    json.dump(summ, open(os.path.join(RES, "h2_%s_summary.json" % tag), "w"), indent=1)
    log("summary written to results/h2_%s_summary.json" % tag)
