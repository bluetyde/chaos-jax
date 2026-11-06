"""Summarise results/h2_main.jsonl against the published cases. Usage: python experiments/h2_analyze.py results/h2_main.jsonl"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from hyperchaos import search3d as s

path = sys.argv[1]
tot = Counter()
hits = []
for line in open(path):
    r = json.loads(line)
    for k, v in r["counts"].items():
        if k != "secs":
            tot[k] += v
    hits.extend(r["hits"])
print("totals", dict(tot), "worker CPU-ish secs sum", None)
pub = {}
for case in "ABCDEFGHIJKLMNOPQRS":
    pub[s.canonical_support(s.support_of(s.case_coefficients(case)))] = case
by_class = defaultdict(list)
for h in hits:
    by_class[tuple(h["canon"])].append(h)
print("confirmed hits", len(hits), "distinct support classes", len(by_class))
print("hits by k", dict(Counter(h["k"] for h in hits)))
print("hit rate per draw %.2e" % (len(hits) / tot["draws"]))
found = {}
for cls, case in pub.items():
    found[case] = len(by_class.get(cls, []))
print("published classes re-found (hits):", found)
print("classes re-found:", sum(1 for v in found.values() if v > 0), "of 19")
new = {c: v for c, v in by_class.items() if c not in pub}
print("distinct classes not in Table I:", len(new), "with hits:", sorted((len(v) for v in new.values()), reverse=True)[:15])
NAMES = s.NAMES


def show(canon, c):
    eqs = []
    for e, v in enumerate("xyz"):
        terms = []
        for m in range(10):
            val = c[10 * e + m]
            if val:
                terms.append("%+g%s" % (val, NAMES[m] if NAMES[m] != "1" else ""))
        eqs.append("%s'=%s" % (v, "".join(terms)))
    return ", ".join(eqs)


# examples of classes not in Table I, strongest hits first
rows = []
for cls, hs in new.items():
    best = max(hs, key=lambda h: h["exps"][0])
    rows.append((len(hs), best["k"], best["exps"], show(cls, best["c"])))
rows.sort(key=lambda r: -r[0])
for n, k, ex, txt in rows[:25]:
    print(n, "hits k=%d" % k, np.round(ex, 3).tolist(), txt)
json.dump(dict(totals=dict(tot), n_hits=len(hits), n_classes=len(by_class), found=found,
               new_classes=[dict(n=r[0], k=r[1], exps=r[2], eq=r[3]) for r in rows]),
          open(path.replace(".jsonl", "_summary.json"), "w"), indent=1)
