"""Re-score stored hits under H1 Amendment 5 (exponent sum above 0.005 means 'expanding'); writes results/h2_<tag>_rescored.json. Usage: python experiments/h2_rescore_amend5.py"""
import glob, json, os
import numpy as np
from hyperchaos import search3d as s
from hyperchaos.classify import CONSERVATIVE_SUM

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
sets = {"night1": [h for p in sorted(glob.glob(os.path.join(RES, "h2_night1_b*.json"))) for h in json.load(open(p))["hits"]],
        "main": [h for line in open(os.path.join(RES, "h2_main.jsonl")) for h in json.loads(line)["hits"]]}
out = {}
for tag, hits in sets.items():
    sums = np.array([sum(h["exps"]) for h in hits])
    removed = [i for i, v in enumerate(sums) if v > CONSERVATIVE_SUM]
    kept_bad = [i for i, v in enumerate(sums) if i not in set(removed) and v > CONSERVATIVE_SUM]
    reconf = []
    for i in removed:
        ok, _ = s.confirm(np.array(hits[i]["c"]))
        reconf.append(bool(ok))
    out[tag] = dict(total=len(hits), removed=len(removed), kept=len(hits) - len(removed), removed_idx=removed, removed_sums_min=float(sums[removed].min()) if removed else None,
                    kept_sum_max=float(max(v for i, v in enumerate(sums) if i not in set(removed))), removed_confirm_true=int(sum(reconf)), kept_bad=len(kept_bad),
                    removed_k={k: sum(1 for i in removed if hits[i]["k"] == k) for k in (5, 6)})
    print(tag, {k: v for k, v in out[tag].items() if k != "removed_idx"}, flush=True)
json.dump(out, open(os.path.join(RES, "h2_amend5_rescored.json"), "w"), indent=1)
