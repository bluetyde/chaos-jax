"""H3 start: four-term 4D search and its controls (docs/PREDICTIONS-H3-FOURTERM-4D.md).
Usage: python experiments/h3_fourterm4d.py c2                 (C2: Li-Sprott exponents through the 4D flow function)
       python experiments/h3_fourterm4d.py c5                 (C5: the screening tiers keep Li-Sprott and Rossler 1979)
       python experiments/h3_fourterm4d.py search <workers>   -> results/h3_fourterm4d_search.json"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
G1 = [round(-3.0 + 0.25 * i, 2) for i in range(25)]
G2 = [round(-3.0 + 0.5 * i, 1) for i in range(13)]


@lru_cache(maxsize=None)
def _stage0(start, dt=0.02, steps_a=1600, steps_b=400):
    import jax
    import jax.numpy as jnp
    from jax import lax

    from hyperchaos import enum4 as e4

    def rk4(f, x):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def one(c):
        f = lambda x: e4.flow4(x, c)  # noqa: E731

        def sa(carry, _):
            x, m = carry
            x = rk4(f, x)
            return (x, jnp.maximum(m, jnp.max(jnp.abs(x)))), None

        def sb(carry, _):
            x, m, lo, hi = carry
            x = rk4(f, x)
            return (x, jnp.maximum(m, jnp.max(jnp.abs(x))), jnp.minimum(lo, x), jnp.maximum(hi, x)), None

        x0 = jnp.asarray(np.array(start))
        (x, m), _ = lax.scan(sa, (x0, jnp.zeros(())), None, length=steps_a)
        (x, m, lo, hi), _ = lax.scan(sb, (x, m, x, x), None, length=steps_b)
        return (m <= 1e4) & (jnp.max(hi - lo) > 1e-3)

    return jax.jit(jax.vmap(one))


def stage0(cs, start=None, batch=1024):
    cs = np.asarray(cs)
    keep = np.zeros(len(cs), dtype=bool)
    for i in range(0, len(cs), batch):
        part = cs[i : i + batch]
        pad = np.zeros((batch, cs.shape[1]))
        pad[: len(part)] = part
        keep[i : i + len(part)] = np.asarray(_stage0(tuple(float(v) for v in (start if start is not None else __import__('hyperchaos').enum4.START4)))(pad))[: len(part)]
    return keep


def tier1_keep(c, x0=None, dt=0.01):
    from hyperchaos import enum4 as e4
    from hyperchaos.classify import classify
    from hyperchaos.spectrum import spectrum_p

    r = spectrum_p(e4.flow4, np.asarray(c), e4.START4 if x0 is None else x0, dt=dt, T=300.0, T_spin=100.0)
    if classify(r, rule="v2").label in ("diverges", "fixed point"):
        return False
    return bool(r.exponents[0] + 3.0 * r.se[0] > 0.005)


def evaluate4(c, x0, T, dt=0.01):
    from hyperchaos import enum4 as e4
    from hyperchaos.classify import classify
    from hyperchaos.spectrum import spectrum_p

    r = spectrum_p(e4.flow4, np.asarray(c), x0, dt=dt, T=T)
    return r, classify(r, rule="v2")


def second_start(x0):
    return np.asarray(x0) * 1.1 + 0.01 * np.array([1.0, -1.0, 1.0, -1.0])


def confirm4(c, x0=None, T=20000.0):
    """Label (chaotic or hyperchaotic, not conservative) agreed from x0 and from a second start, and the first start's spectrum; label None if not confirmed (Amendment A1)."""
    from hyperchaos import enum4 as e4

    x0 = e4.START4 if x0 is None else x0
    r1, k1 = evaluate4(c, x0, T)
    if k1.label not in ("chaotic", "hyperchaotic") or k1.conservative:
        return None, r1
    r2, k2 = evaluate4(c, second_start(x0), T)
    if k2.label == k1.label and not k2.conservative:
        return k1.label, r1
    return None, r1


def pick_start(c, keeps, j):
    """First start (in e4.STARTS4 order) that passed stage 0 for system j and also passes tier 1; None if there is none."""
    from hyperchaos import enum4 as e4

    for si, x0 in enumerate(e4.STARTS4):
        if keeps[si][j] and tier1_keep(c, x0):
            return x0
    return None


def w_chunk(args):
    sups, grid_by_d = args
    from hyperchaos import enum4 as e4

    n = s0 = t1 = st2 = 0
    hits, cons = [], []
    by_rank = {}
    for sup in sups:
        d = e4.null_directions(sup)[0].shape[1]
        r = e4.null_directions(sup)[1]
        cs = e4.reps(sup, grid_by_d[d] if d else [0.0])
        n += len(cs)
        by_rank[r] = by_rank.get(r, 0) + len(cs)
        keeps = [stage0(cs, x0) for x0 in e4.STARTS4]
        keep = np.any(keeps, axis=0)
        s0 += int(keep.sum())
        for j in np.flatnonzero(keep):
            x0 = pick_start(cs[j], keeps, j)
            if x0 is None:
                continue
            t1 += 1
            rr, kk = evaluate4(cs[j], x0, 5000.0)
            if kk.label in ("chaotic", "hyperchaotic") and kk.conservative:
                cons.append(dict(c=cs[j].tolist(), rank=r, label=kk.label, exps=rr.exponents.tolist(), se=rr.se.tolist(), support=list(sup)))
                continue
            if kk.label not in ("chaotic", "hyperchaotic"):
                continue
            st2 += 1
            lab, r5 = confirm4(cs[j], x0)
            if lab is not None:
                hits.append(dict(c=cs[j].tolist(), rank=r, label=lab, exps=r5.exponents.tolist(), se=r5.se.tolist(), support=list(sup), start=np.asarray(x0).tolist()))
    return dict(n=n, stage0=s0, tier1=t1, stage2=st2, hits=hits, cons=cons, by_rank=by_rank)


def merge(parts):
    out = dict(n=0, stage0=0, tier1=0, stage2=0, hits=[], cons=[], by_rank={})
    for p in parts:
        for k in ("n", "stage0", "tier1", "stage2"):
            out[k] += p[k]
        out["hits"] += p["hits"]
        out["cons"] += p["cons"]
        for r, v in p["by_rank"].items():
            out["by_rank"][r] = out["by_rank"].get(r, 0) + v
    return out


def c_from_system(name):
    from hyperchaos import enum4 as e4

    c = np.zeros(e4.N * e4.NM)

    def one(i):
        return tuple(1 if j == i else 0 for j in range(4))

    def two(i, j):
        return tuple((1 if k == i else 0) + (1 if k == j else 0) for k in range(4))

    zero = (0, 0, 0, 0)
    if name == "LS":
        terms = [(0, one(1), 1.0), (0, one(0), -1.0), (1, two(0, 2), -1.0), (1, one(3), 1.0), (2, two(0, 1), 1.0), (2, zero, -2.6), (3, one(1), -0.44)]
    else:  # Rossler 1979, a = 0.25, b = 3, c = 0.5, d = 0.05
        terms = [(0, one(1), -1.0), (0, one(2), -1.0), (1, one(0), 1.0), (1, one(1), 0.25), (1, one(3), 1.0), (2, zero, 3.0), (2, two(0, 2), 1.0), (3, one(2), -0.5), (3, one(3), 0.05)]
    for eq, mono, v in terms:
        c[e4.NM * eq + e4._MIDX[mono]] = v
    return c


if __name__ == "__main__":
    mode = sys.argv[1]
    from hyperchaos import enum4 as e4

    t0 = time.time()
    if mode == "c2":
        from hyperchaos.spectrum import spectrum_p

        r = spectrum_p(e4.flow4, c_from_system("LS"), np.array([1.0, 1.0, 1.0, 1.0]), T=20000.0)
        pub = np.array([0.0704, 0.0128, 0.0, -1.0832])
        print("C2 exponents", np.round(r.exponents, 4).tolist(), "published", pub.tolist(), "max diff %.4f" % np.max(np.abs(r.exponents - pub)))
        json.dump(dict(exps=r.exponents.tolist(), published=pub.tolist()), open(os.path.join(RES, "h3_c2.json"), "w"))
    elif mode == "c5":
        out = {}
        for name in ("LS", "R79"):
            c = c_from_system(name)
            keeps = [stage0(np.array([c]), x0) for x0 in e4.STARTS4]
            s0 = bool(np.any(keeps))
            x0 = pick_start(c, keeps, 0) if s0 else None
            lab, r = confirm4(c, x0, 20000.0) if x0 is not None else (None, None)
            out[name] = dict(stage0=s0, tier1=x0 is not None, start=None if x0 is None else np.asarray(x0).tolist(), label=lab, exps=None if r is None else r.exponents.tolist())
            print(name, out[name], flush=True)
        json.dump(out, open(os.path.join(RES, "h3_c5.json"), "w"), indent=1)
    else:
        workers = int(sys.argv[2])
        sups = e4.supports4(4)
        chunks = [(sups[i : i + 10], {1: G1, 2: G2, 3: G2}) for i in range(0, len(sups), 10)]
        print("support classes", len(sups), "chunks", len(chunks), flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = []
            for i, p in enumerate(ex.map(w_chunk, chunks)):
                parts.append(p)
                if i % 10 == 0:
                    m = merge(parts)
                    print(i, "/", len(chunks), {k: v for k, v in m.items() if k not in ("hits", "cons")}, "hits", len(m["hits"]), "cons", len(m["cons"]), "elapsed %.0f s" % (time.time() - t0), flush=True)
        out = merge(parts)
        out["wall"] = time.time() - t0
        json.dump(out, open(os.path.join(RES, "h3_fourterm4d_search.json"), "w"))
        print({k: v for k, v in out.items() if k not in ("hits", "cons")}, "hits", len(out["hits"]), "cons", len(out["cons"]))
    print("wall %.0f s" % (time.time() - t0))
