"""Night run analysis: dedup hits by the 48-element symmetry group, split by k, compare with the 1e7 main run. Usage: python experiments/h2_night_analyze.py night1"""
import glob, json, os, sys
from collections import Counter
import numpy as np
from hyperchaos import orbits

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
tag = sys.argv[1]
hits = [h for p in sorted(glob.glob(os.path.join(RES, "h2_%s_b*.json" % tag))) for h in json.load(open(p))["hits"]]
key = lambda c: tuple(np.round(orbits.canonical(np.array(c)), 3).tolist())
keys = [key(h["c"]) for h in hits]
cnt = Counter(keys)
print("hits", len(hits), "distinct classes", len(cnt), "classes seen more than once", sum(1 for v in cnt.values() if v > 1), "max multiplicity", max(cnt.values()))
print("by k:", dict(Counter(h["k"] for h in hits)))
print("distinct classes by k:", dict(Counter(k for k in {key_: h["k"] for key_, h in zip(keys, hits)}.values())))
main = [h for line in open(os.path.join(RES, "h2_main.jsonl")) for h in json.loads(line)["hits"]]
mk = {key(h["c"]) for h in main}
print("main run hits", len(main), "classes", len(mk), "of these classes also found tonight", len(mk & set(cnt)))
l1 = np.array([h["exps"][0] for h in hits]); print("lambda1 median %.3f max %.3f" % (np.median(l1), l1.max()))
print("hyperchaotic (lambda2>0.01):", sum(1 for h in hits if h["exps"][1] > 0.01))
json.dump(dict(hits=len(hits), classes=len(cnt), multi=sum(1 for v in cnt.values() if v > 1)), open(os.path.join(RES, "h2_%s_analysis.json" % tag), "w"))
