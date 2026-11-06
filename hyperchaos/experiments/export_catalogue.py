"""Export the verified 3D quadratic chaotic flows found by the H2 searches to catalogue/.

Inputs: results/h2_night1_b*.json (night run 1) and results/h2_main.jsonl (main run), minus the hits
that H1 Amendment 5 removes (positive exponent sum, see results/h2_amend5_rescored.json).
Outputs: catalogue/hits.csv (every kept hit), catalogue/families.csv (one row per structure key,
the representative with the smallest total standard error), catalogue/catalogue.json (families with metadata).
"Chaotic" here means: passed the evaluator's checks with a positive largest exponent, a near-zero
second exponent and a negative third (numerical evidence, not a proof). Nothing here is claimed new."""
import csv, glob, json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hyperchaos import keys
from hyperchaos.search3d import NAMES

RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "catalogue")
os.makedirs(OUT, exist_ok=True)

sets = {"night1": [h for p in sorted(glob.glob(os.path.join(RES, "h2_night1_b*.json"))) for h in json.load(open(p))["hits"]],
        "main": [h for line in open(os.path.join(RES, "h2_main.jsonl")) for h in json.loads(line)["hits"]]}
removed = json.load(open(os.path.join(RES, "h2_amend5_rescored.json")))


def equations(c):
    out = []
    for e, var in enumerate("xyz"):
        terms = []
        for m in range(10):
            v = c[10 * e + m]
            if v != 0:
                terms.append(f"{v:+g}" + ("" if NAMES[m] == "1" else NAMES[m]))
        out.append(f"d{var}/dt = " + (" ".join(terms) if terms else "0"))
    return "; ".join(out)


def ky_dim(ex):
    ex = sorted(ex, reverse=True)
    s = 0.0
    for j, v in enumerate(ex):
        if s + v < 0:
            return j + (s / abs(v))
        s += v
    return float(len(ex))


rows = []
for tag, hits in sets.items():
    rm = set(removed[tag]["removed_idx"])
    for i, h in enumerate(hits):
        if i in rm:
            continue
        ex = h["exps"]
        rows.append(dict(source=tag, index=i, terms=h["k"], equations=equations(h["c"]),
                         l1=ex[0], l2=ex[1], l3=ex[2], exp_sum=sum(ex), ky_dim=ky_dim(ex),
                         se1=h["se"][0], se2=h["se"][1], se3=h["se"][2],
                         structure_key=json.dumps(list(keys.structure_key(h["c"]))), coeffs=json.dumps(h["c"])))

fields = list(rows[0])
with open(os.path.join(OUT, "hits.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fields)
    w.writeheader()
    w.writerows(rows)

fam = {}
for r in rows:
    k = r["structure_key"]
    fam.setdefault(k, []).append(r)
fam_rows = []
for k, rs in fam.items():
    best = min(rs, key=lambda r: r["se1"] + r["se2"] + r["se3"])
    fam_rows.append(dict(best, n_hits=len(rs)))
fam_rows.sort(key=lambda r: (r["terms"], -r["l1"]))
with open(os.path.join(OUT, "families.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, list(fam_rows[0]))
    w.writeheader()
    w.writerows(fam_rows)
json.dump(fam_rows, open(os.path.join(OUT, "catalogue.json"), "w"), indent=1)
print(dict(hits=len(rows), families=len(fam_rows), by_terms={k: sum(1 for r in fam_rows if r["terms"] == k) for k in sorted({r["terms"] for r in fam_rows})}))
