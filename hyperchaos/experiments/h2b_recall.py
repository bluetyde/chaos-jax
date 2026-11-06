"""Gate GA: put every confirmed hit of results/h2_main.jsonl through the middle filter alone.
Usage: python experiments/h2b_recall.py <workers>   (writes results/h2b_recall.json)"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def w(args):
    i, c = args
    from hyperchaos import search3d as s
    from hyperchaos.spectrum import spectrum_p

    c = np.array(c)
    kept = s.mid_filter(c)
    tiers = []
    for T, spin in s.MID_TIERS:
        r = spectrum_p(s.flow, c, s.START, dt=0.01, T=T, T_spin=spin)
        tiers.append(dict(T=T, lam1=float(r.exponents[0]), se1=float(r.se[0]), drop=bool(s.mid_drop(r))))
    return dict(i=i, kept=bool(kept), tiers=tiers)


if __name__ == "__main__":
    hits = []
    for line in open(os.path.join(RES, "h2_main.jsonl")):
        hits.extend(json.loads(line)["hits"])
    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as ex:
        out = list(ex.map(w, [(i, h["c"]) for i, h in enumerate(hits)], chunksize=8))
    kept = sum(o["kept"] for o in out)
    lost = [dict(i=o["i"], k=hits[o["i"]]["k"], exps=hits[o["i"]]["exps"], tiers=o["tiers"]) for o in out if not o["kept"]]
    print("hits", len(hits), "kept", kept, "recall %.4f" % (kept / len(hits)), "lost", len(lost))
    for l in lost:
        print(l["i"], "k", l["k"], np.round(l["exps"], 4).tolist(), [(t["T"], round(t["lam1"], 4), round(t["se1"], 4), t["drop"]) for t in l["tiers"]])
    json.dump(dict(n=len(hits), kept=kept, lost=lost), open(os.path.join(RES, "h2b_recall.json"), "w"), indent=1)
