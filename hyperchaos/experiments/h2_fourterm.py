"""H2 four-term search and its rediscovery control (docs/PREDICTIONS-H2-FOURTERM.md).
Usage: python experiments/h2_fourterm.py control <workers>   -> results/h2_fourterm_control.json
       python experiments/h2_fourterm.py search <workers>    -> results/h2_fourterm_search.json"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
COARSE = [round(-3.0 + 0.5 * i, 1) for i in range(13)]
FINE = [round(-3.0 + 0.1 * i, 1) for i in range(61)]


def w_chunk(args):
    sups, grid = args
    from hyperchaos import enumerate as en
    from hyperchaos import search3d as s

    n = ka = kb = bd = st2 = 0
    hits = []
    cons = []
    by_rank = {}
    for sup in sups:
        cs = en.reps(sup, grid)
        r = en.null_directions(sup)[1]
        idx, lam, keep_a = s.screen(cs)
        n += len(cs)
        ka += int(keep_a.sum())
        kb += len(idx)
        if len(idx):
            idx = idx[s.bound_keep(cs[idx])]
        bd += len(idx)
        for j in idx:
            r2, k2 = s.evaluate(cs[j], s.START, 5000.0)
            if k2.label == "chaotic" and k2.conservative:  # chaotic with exponent sum near zero: not a dissipative attractor, recorded separately
                cons.append(dict(c=cs[j].tolist(), rank=r, exps=r2.exponents.tolist(), se=r2.se.tolist(), support=list(sup)))
                continue
            ok, rr = s.is_chaotic(cs[j], 5000.0)
            if not ok:
                continue
            st2 += 1
            ok3, r3 = s.confirm(cs[j], 20000.0)
            if ok3:
                hits.append(dict(c=cs[j].tolist(), rank=r, exps=r3.exponents.tolist(), se=r3.se.tolist(), support=list(sup)))
        by_rank[r] = by_rank.get(r, 0) + len(cs)
    return dict(n=n, keep_a=ka, keep_b=kb, bound=bd, stage2=st2, hits=hits, cons=cons, by_rank=by_rank)


def merge(parts):
    out = dict(n=0, keep_a=0, keep_b=0, bound=0, stage2=0, hits=[], cons=[], by_rank={})
    for p in parts:
        for k in ("n", "keep_a", "keep_b", "bound", "stage2"):
            out[k] += p[k]
        out["hits"] += p["hits"]
        out["cons"] += p["cons"]
        for r, v in p["by_rank"].items():
            out["by_rank"][r] = out["by_rank"].get(r, 0) + v
    return out


if __name__ == "__main__":
    mode, workers = sys.argv[1], int(sys.argv[2])
    from hyperchaos import enumerate as en
    from hyperchaos import search3d as s

    t0 = time.time()
    if mode == "control":
        cases = sys.argv[3].split(",") if len(sys.argv) > 3 else ["A", "B", "C", "D", "E"]
        sups = {k: tuple(sorted(s.support_of(s.case_coefficients(k)))) for k in cases}
        for k in cases:
            inv, r = en.invariants(s.case_coefficients(k))
            print("case", k, "support", sups[k], "rank", r, "invariant(s)", np.round(inv, 3).tolist(), "in range" if np.all(np.abs(inv) <= 3.0) else "OUT OF RANGE", flush=True)
        with ProcessPoolExecutor(max_workers=min(workers, len(cases))) as ex:
            res = dict(zip(cases, ex.map(w_chunk, [([sups[k]], FINE) for k in cases])))
        out = {k: dict(found=len(v["hits"]) > 0, found_conservative=len(v["cons"]) > 0, n=v["n"], keep_a=v["keep_a"], stage2=v["stage2"], hits=len(v["hits"]), conservative_hits=len(v["cons"]), hit_example=(v["hits"] or v["cons"])[:1]) for k, v in res.items()}
        for k, v in out.items():
            print(k, {a: b for a, b in v.items() if a != "hit_example"}, flush=True)
        print("found dissipative", sum(v["found"] for v in out.values()), "conservative", sum(v["found_conservative"] for v in out.values()), "of", len(cases))
        json.dump(out, open(os.path.join(RES, "h2_fourterm_control_%s.json" % "".join(cases)), "w"), indent=1)
    else:
        sups = en.supports(4)
        chunks = [(sups[i : i + 40], COARSE) for i in range(0, len(sups), 40)]
        print("supports", len(sups), "chunks", len(chunks), flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = []
            for i, p in enumerate(ex.map(w_chunk, chunks)):
                parts.append(p)
                if i % 20 == 0:
                    m = merge(parts)
                    print(i, "/", len(chunks), {k: v for k, v in m.items() if k != "hits"}, "hits", len(m["hits"]), "elapsed %.0f s" % (time.time() - t0), flush=True)
        out = merge(parts)
        out["wall"] = time.time() - t0
        json.dump(out, open(os.path.join(RES, "h2_fourterm_search.json"), "w"))
        print({k: v for k, v in out.items() if k != "hits"}, "hits", len(out["hits"]), "conservative", len(out["cons"]))
    print("wall %.0f s" % (time.time() - t0))
