"""Where does one 20000-draw chunk spend its time, and how do stage-1b survivors fail? Usage: python experiments/h2b_profile.py <seed>"""
import sys
import time
from collections import Counter

import numpy as np

from hyperchaos import search3d as s
from hyperchaos.classify import classify
from hyperchaos.spectrum import spectrum_p

seed = int(sys.argv[1])
rng = np.random.default_rng(seed)
cs = s.sample(rng, 20000)
s.screen(cs[:2048])  # compile
t = time.time(); idx, lam, ka = s.screen(cs); t_screen = time.time() - t
t = time.time(); mids = [s.mid_filter(cs[i]) for i in idx]; t_mid = time.time() - t
t = time.time()
labels = Counter()
for i in idx:
    r = spectrum_p(s.flow, cs[i], s.START, dt=0.01, T=5000.0)
    labels[classify(r, rule="v2").label + ("+cons" if classify(r, rule="v2").conservative else "")] += 1
t_stage2 = time.time() - t
print("survivors of 1b", len(idx), "kept by mid", sum(mids))
print("seconds: screen %.1f, mid (on all 1b survivors) %.1f, stage 2 (T=5000, on all 1b survivors) %.1f" % (t_screen, t_mid, t_stage2))
print("stage-2 labels of 1b survivors", dict(labels))
