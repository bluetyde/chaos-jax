"""H2F W1a: orbit duplicates among the 472 confirmed hits of results/h2_main.jsonl. Usage: python experiments/h2f_orbits.py"""
import json
import os
from collections import Counter

import numpy as np

from hyperchaos import orbits

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
hits = []
for line in open(os.path.join(RES, "h2_main.jsonl")):
    r = json.loads(line)
    hits.extend((r["seed"], h) for h in r["hits"])
canon = [orbits.canonical(np.array(h["c"])) for _, h in hits]
cnt = Counter(canon)
dups = {k: v for k, v in cnt.items() if v > 1}
stab = [len({tuple(np.round(v, 6) + 0.0) for v in orbits.images(np.array(h["c"]))}) for _, h in hits]
print("hits", len(hits), "distinct orbits", len(cnt), "orbits with more than one hit", len(dups))
print("orbit size (distinct images) over hits:", dict(Counter(stab)))
out = dict(n_hits=len(hits), n_orbits=len(cnt), duplicates=[dict(count=v, seeds=[hits[i][0] for i, c in enumerate(canon) if c == k]) for k, v in dups.items()],
           orbit_size_counts={str(k): v for k, v in Counter(stab).items()})
json.dump(out, open(os.path.join(RES, "h2f_orbits.json"), "w"), indent=1)
