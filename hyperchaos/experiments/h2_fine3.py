"""Fine invariant-grid pass on the rank-3 four-term supports (docs/PREDICTIONS-H2-FINE3.md). Resumable: one file per chunk in results/h2_fine3/.
Usage: python -u experiments/h2_fine3.py <workers>"""
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h2_fourterm import FINE, merge, w_chunk  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "h2_fine3")
STALL = float(os.environ.get("H2_STALL_S", "1800"))
PER = 10


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


if __name__ == "__main__":
    workers = int(sys.argv[1])
    from hyperchaos import enumerate as en

    os.makedirs(RES, exist_ok=True)
    sups = [sp for sp in en.supports(4) if en.null_directions(sp)[1] == 3]
    chunks = [sups[i : i + PER] for i in range(0, len(sups), PER)]
    todo = [i for i in range(len(chunks)) if not os.path.exists(os.path.join(RES, "chunk_%03d.json" % i))]
    log("rank-3 supports %d, chunks %d, already done %d" % (len(sups), len(chunks), len(chunks) - len(todo)))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        pending = {}
        it = iter(todo)
        for _ in range(workers * 2):
            i = next(it, None)
            if i is not None:
                pending[ex.submit(w_chunk, (chunks[i], FINE))] = i
        last = time.time()
        done_n = 0
        while pending:
            fin, _ = wait(list(pending), timeout=60, return_when=FIRST_COMPLETED)
            if not fin and time.time() - last > STALL:
                log("STOP: no chunk finished for %.0f s; killing the worker processes" % STALL)
                for p in list(ex._processes.values()):
                    p.terminate()
                ex.shutdown(wait=False, cancel_futures=True)
                sys.exit(2)
            for f in fin:
                i = pending.pop(f)
                r = f.result()
                tmp = os.path.join(RES, "chunk_%03d.json.tmp" % i)
                json.dump(r, open(tmp, "w"))
                os.replace(tmp, os.path.join(RES, "chunk_%03d.json" % i))
                last = time.time()
                done_n += 1
                if done_n % 10 == 0:
                    log("%d/%d chunks this run, hits %d, cons %d, elapsed %.0f min" % (done_n, len(todo), len(r["hits"]), len(r["cons"]), (time.time() - t0) / 60))
                j = next(it, None)
                if j is not None:
                    pending[ex.submit(w_chunk, (chunks[j], FINE))] = j
    parts = [json.load(open(os.path.join(RES, "chunk_%03d.json" % i))) for i in range(len(chunks))]
    out = merge(parts)
    out["wall_this_run_s"] = time.time() - t0
    json.dump(out, open(os.path.join(os.path.dirname(RES), "h2_fine3_summary.json"), "w"))
    log("done: %s hits %d cons %d" % ({k: v for k, v in out.items() if k not in ("hits", "cons")}, len(out["hits"]), len(out["cons"])))
