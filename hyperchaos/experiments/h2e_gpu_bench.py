"""H2E GPU benchmark: float32 Lyapunov spectrum of the 3D quadratic family in PyTorch (CUDA graph), same method as spectrum32.py
(RK4 on state + tangent frame, Gram-Schmidt re-orthonormalisation every 10 steps, spin 200). Benchmark only.
Run with the torch venv:  <python> experiments/h2e_gpu_bench.py <T> <batch> [<batch> ...]
Memory is capped at 25% of the card and a watchdog exits above 8.5 GB total use (the card also drives the displays)."""
import json
import os
import sys
import threading
import time

import numpy as np
import torch

CAP = 0.25
LIMIT = 8.5e9
torch.cuda.set_per_process_memory_fraction(CAP)


def watchdog():
    while True:
        free, total = torch.cuda.mem_get_info()
        if total - free > LIMIT:
            print("WATCHDOG: GPU memory above limit, exiting", flush=True)
            os._exit(3)
        time.sleep(0.5)


threading.Thread(target=watchdog, daemon=True).start()
dev = "cuda"
DT = 0.01
QR_EVERY = 10
START = (0.05, 0.05, 0.05)

# D(x) = D0 + X Dx + Y Dy + Z Dz : derivative of the 10 monomials [1,x,y,z,xx,xy,xz,yy,yz,zz] w.r.t. (x,y,z), as (10,3) matrices
D = np.zeros((4, 10, 3), dtype=np.float32)
D[0, 1, 0] = D[0, 2, 1] = D[0, 3, 2] = 1.0
D[1, 4, 0] = 2.0; D[2, 5, 0] = 1.0; D[1, 5, 1] = 1.0; D[3, 6, 0] = 1.0; D[1, 6, 2] = 1.0
D[2, 7, 1] = 2.0; D[3, 8, 1] = 1.0; D[2, 8, 2] = 1.0; D[3, 9, 2] = 2.0
Dt = torch.tensor(D, device=dev)  # (4,10,3)


def build(cs, T, spin=200.0):
    B = len(cs)
    C = torch.tensor(np.asarray(cs, dtype=np.float32).reshape(B, 3, 10), device=dev)
    CD = torch.einsum("bim,kmj->bkij", C, Dt)  # (B,4,3,3): J = sum_k w_k CD_k with w = [1,x,y,z]
    x = torch.tensor(START, device=dev).repeat(B, 1)
    q = torch.eye(3, device=dev).repeat(B, 1, 1)
    ssum = torch.zeros(B, 3, device=dev)
    div = torch.zeros(B, device=dev)
    mx = torch.zeros(B, device=dev)

    def aug(x, q):
        X, Y, Z = x[:, 0], x[:, 1], x[:, 2]
        feats = torch.stack([torch.ones_like(X), X, Y, Z, X * X, X * Y, X * Z, Y * Y, Y * Z, Z * Z], 1)
        f = torch.bmm(C, feats.unsqueeze(2)).squeeze(2)
        w = torch.stack([torch.ones_like(X), X, Y, Z], 1)
        J = torch.einsum("bk,bkij->bij", w, CD)
        return f, torch.bmm(J, q), J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]

    def rk4_state(x):
        k1 = aug(x, q)[0]
        k2 = aug(x + 0.5 * DT * k1, q)[0]
        k3 = aug(x + 0.5 * DT * k2, q)[0]
        k4 = aug(x + DT * k3, q)[0]
        return x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def chunk():
        nonlocal q
        xx, qq, d = x, q, torch.zeros_like(div)
        for _ in range(QR_EVERY):
            k1x, k1q, t1 = aug(xx, qq)
            k2x, k2q, t2 = aug(xx + 0.5 * DT * k1x, qq + 0.5 * DT * k1q)
            k3x, k3q, t3 = aug(xx + 0.5 * DT * k2x, qq + 0.5 * DT * k2q)
            k4x, k4q, t4 = aug(xx + DT * k3x, qq + DT * k3q)
            xx = xx + DT / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
            qq = qq + DT / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
            d = d + DT / 6.0 * (t1 + 2 * t2 + 2 * t3 + t4)
        v0, v1, v2 = qq[:, :, 0], qq[:, :, 1], qq[:, :, 2]
        n0 = v0.norm(dim=1, keepdim=True); u0 = v0 / n0
        w1 = v1 - (u0 * v1).sum(1, keepdim=True) * u0; n1 = w1.norm(dim=1, keepdim=True); u1 = w1 / n1
        w2 = v2 - (u0 * v2).sum(1, keepdim=True) * u0 - (u1 * v2).sum(1, keepdim=True) * u1; n2 = w2.norm(dim=1, keepdim=True); u2 = w2 / n2
        x.copy_(xx)
        q.copy_(torch.stack([u0, u1, u2], 2))
        ssum.add_(torch.log(torch.cat([n0, n1, n2], 1)))
        div.add_(d)
        mx.copy_(torch.maximum(mx, xx.abs().amax(1)))

    return dict(B=B, x=x, q=q, ssum=ssum, div=div, mx=mx, chunk=chunk, rk4_state=rk4_state, spin=int(round(spin / DT)), n_chunks=int(round(T / (QR_EVERY * DT))))


def run(cs, T, graph=True):
    s = build(cs, T)
    torch.cuda.synchronize()
    t0 = time.time()
    # spin-up on state only (eager loop in a graph of 100 steps)
    def spin_block():
        xx = s["x"]
        for _ in range(100):
            xx = s["rk4_state"](xx)
        s["x"].copy_(xx)

    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        spin_block(); s["chunk"]()  # warm-up (alters state; reset below)
    torch.cuda.current_stream().wait_stream(st)
    s["x"].copy_(torch.tensor(START, device=dev).repeat(s["B"], 1)); s["q"].copy_(torch.eye(3, device=dev).repeat(s["B"], 1, 1))
    s["ssum"].zero_(); s["div"].zero_(); s["mx"].zero_()
    g_spin, g_chunk = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_spin):
        spin_block()
    with torch.cuda.graph(g_chunk):
        s["chunk"]()
    s["x"].copy_(torch.tensor(START, device=dev).repeat(s["B"], 1)); s["q"].copy_(torch.eye(3, device=dev).repeat(s["B"], 1, 1))
    s["ssum"].zero_(); s["div"].zero_(); s["mx"].zero_()
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(s["spin"] // 100):
        g_spin.replay()
    for _ in range(s["n_chunks"]):
        g_chunk.replay()
    torch.cuda.synchronize()
    t2 = time.time()
    tt = s["n_chunks"] * QR_EVERY * DT
    exps = (s["ssum"] / tt).sort(dim=1, descending=True).values.cpu().numpy()
    return exps, (s["div"] / tt).cpu().numpy(), s["mx"].cpu().numpy(), dict(setup_s=t1 - t0, run_s=t2 - t1, per_chunk_ms=1000 * (t2 - t1) / max(1, s["n_chunks"] + s["spin"] // 1000))


if __name__ == "__main__":
    T = float(sys.argv[1])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hits = [h for l in open(os.path.join(root, "results", "h2_main.jsonl")) for h in json.loads(l)["hits"]]
    pool = np.array([h["c"] for h in hits])
    out = []
    if os.environ.get("CASES"):  # validation against named published cases: CASES=BSFO, coefficients from results/h2e_cases.npy
        pool = np.load(os.path.join(root, "results", "h2e_cases.npy"))
    for B in map(int, sys.argv[2:]):
        cs = pool[np.arange(B) % len(pool)]
        exps, dv, mx, tm = run(cs, T)
        free, total = torch.cuda.mem_get_info()
        r = dict(T=T, B=B, run_s=tm["run_s"], setup_s=tm["setup_s"], s_per_system_at_T=tm["run_s"] / B, s_per_system_extrap_T5000=tm["run_s"] / B * 5000.0 / T,
                 finite_frac=float(np.isfinite(exps).all(1).mean()), gpu_used_mb=(total - free) / 2**20, torch_peak_mb=torch.cuda.max_memory_allocated() / 2**20)
        print(json.dumps(r), flush=True)
        out.append(r)
        if os.environ.get("CASES"):
            for j, k in enumerate(os.environ["CASES"]):
                print(k, np.round(exps[j], 4).tolist(), "div", round(float(dv[j]), 4), flush=True)
        if B == 256:
            np.save(os.path.join(root, "results", "h2e_gpu_check.npy"), np.concatenate([exps[:8], dv[:8, None]], 1))
    json.dump(out, open(os.path.join(root, "results", "h2e_gpu_bench_T%d.json" % int(T)), "w"), indent=1)
