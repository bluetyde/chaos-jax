"""Group the night-run hits by structure key and compare with Sprott's cases A-S. Usage: python experiments/h2_keys_compare.py"""
import glob, json, os
from collections import Counter
import numpy as np
from hyperchaos import keys
from hyperchaos import search3d as s

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
hits = [h for p in sorted(glob.glob(os.path.join(RES, "h2_night1_b*.json"))) for h in json.load(open(p))["hits"] if sum(h["exps"]) <= 0.005]
main = [h for line in open(os.path.join(RES, "h2_main.jsonl")) for h in json.loads(line)["hits"] if sum(h["exps"]) <= 0.005]
print("night hits", len(hits), "main hits", len(main))
sk = [keys.structure_key(h["c"]) for h in hits]
cnt = Counter(sk)
k5 = {k for k, h in zip(sk, hits) if h["k"] == 5}
k6 = {k for k, h in zip(sk, hits) if h["k"] == 6}
print("structure keys (term-set + signs, up to renaming/flips): all %d, 5-term %d, 6-term %d" % (len(cnt), len(k5), len(k6)))
print("hits per structure key: max %d, median %d" % (max(cnt.values()), int(np.median(list(cnt.values())))))
sup = Counter(keys.support_key(h["c"]) for h in hits)
print("term-sets only (signs ignored): %d" % len(sup))
sprott = {k: keys.structure_key(s.case_coefficients(k)) for k in "ABCDEFGHIJKLMNOPQRS"}
night_keys = set(sk)
main_keys = {keys.structure_key(h["c"]) for h in main}
print("Sprott cases A-S found by structure key in the night run:", [k for k, v in sprott.items() if v in night_keys])
print("Sprott cases A-S found in the 1e7 main run:", [k for k, v in sprott.items() if v in main_keys])
print("main-run structure keys %d; shared with night run: %d" % (len(main_keys), len(main_keys & night_keys)))
fam = Counter(zip(sk, [keys.param_key(h["c"]) for h in hits]))
print("distinct (structure, parameter) pairs after rounding to 3 decimals:", len(fam))
json.dump(dict(night_hits=len(hits), structure_keys=len(cnt), keys_5=len(k5), keys_6=len(k6), term_sets=len(sup), sprott_in_night=[k for k, v in sprott.items() if v in night_keys],
               sprott_in_main=[k for k, v in sprott.items() if v in main_keys], main_keys=len(main_keys), shared=len(main_keys & night_keys)), open(os.path.join(RES, "h2_keys_compare.json"), "w"), indent=1)
