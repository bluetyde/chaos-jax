"""H2D timing: float32 batched cost per system at several batch sizes against float64 one at a time, one process, same hits."""
import json, os, time
import numpy as np
from hyperchaos import search3d as s
from hyperchaos.spectrum import spectrum_p
from hyperchaos.spectrum32 import spectrum32_batch
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
hits = [h for l in open(os.path.join(RES, "h2_main.jsonl")) for h in json.loads(l)["hits"]]
cs = np.array([h["c"] for h in hits[:128]])
out = {}
for b in (16, 32, 64, 128):
    spectrum32_batch(s.flow, cs[:b], s.START, batch=b)  # compile
    t = time.process_time(); w = time.time()
    spectrum32_batch(s.flow, cs[:b], s.START, batch=b)
    out["f32_batch%d" % b] = dict(cpu_s_per_system=(time.process_time() - t) / b, wall_s_per_system=(time.time() - w) / b)
    print("float32 batch", b, out["f32_batch%d" % b], flush=True)
spectrum_p(s.flow, cs[0], s.START, dt=0.01, T=5000.0)
t = time.process_time(); w = time.time()
for c in cs[:8]:
    spectrum_p(s.flow, c, s.START, dt=0.01, T=5000.0)
out["f64_sequential"] = dict(cpu_s_per_system=(time.process_time() - t) / 8, wall_s_per_system=(time.time() - w) / 8)
print("float64 one at a time", out["f64_sequential"])
json.dump(out, open(os.path.join(RES, "h2d_timing.json"), "w"), indent=1)
