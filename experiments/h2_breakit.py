"""Planted-bug check for H2 (gate G5). Run only on a clean, committed tree.

Each bug is written into the source, the fast tests are run, and the file is restored with git.
Usage: python experiments/h2_breakit.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

BUGS = [
    (
        "1 support class ignores one of the 6 variable permutations",
        "src/hyperchaos/search3d.py",
        "    perms = list(permutations(range(3))) if perms is None else perms" + chr(10) + "    return min(",
        "    perms = list(permutations(range(3)))[:5] if perms is None else perms" + chr(10) + "    return min(",
    ),
    (
        "2 stage 1a test inverted (drops chaos, keeps the wrong orbits)",
        "src/hyperchaos/search3d.py",
        "        return (m <= 1e4) & moving" + chr(10),
        "        return (m > 1e4) & moving" + chr(10),
    ),
    (
        "3 monomial order swapped in the flow (decode swaps xz and yy)",
        "src/hyperchaos/search3d.py",
        "x[0] * x[2], x[1] * x[1], x[1] * x[2]",
        "x[1] * x[1], x[0] * x[2], x[1] * x[2]",
    ),
    (
        "4 sampler grid back to magnitudes up to 10 (the 2026-10-27 mistake)",
        "src/hyperchaos/search3d.py",
        'os.environ.get("H2_GRID_MAX", "5.0")',
        'os.environ.get("H2_GRID_MAX", "10.0")',
    ),
    (
        "5 middle filter drop test inverted (keeps failing systems, drops chaos)",
        "src/hyperchaos/search3d.py",
        "res.exponents[0] + 3.0 * res.se[0] <= MID_EPS",
        "res.exponents[0] + 3.0 * res.se[0] > MID_EPS",
    ),
    (
        "6 middle filter threshold so low that nothing is dropped on exponents",
        "src/hyperchaos/search3d.py",
        "MID_EPS = 0.005",
        "MID_EPS = -1e9",
    ),
    (
        "7 boundedness filter keeps the escapes and drops the bounded orbits",
        "src/hyperchaos/search3d.py",
        "return (tf < 0) & (rg >= BOUND_STUCK)",
        "return (tf >= 0) & (rg >= BOUND_STUCK)",
    ),
    (
        "8 boundedness filter stuck limit zero (sinks no longer dropped)",
        "src/hyperchaos/search3d.py",
        "BOUND_STUCK = 1e-3",
        "BOUND_STUCK = 0.0",
    ),
    (
        "9 float32 filter result inverted",
        "src/hyperchaos/search3d.py",
        "        ok &= np.abs(e).min(1) <= F32_ZERO" + chr(10) + "    return ok",
        "        ok &= np.abs(e).min(1) <= F32_ZERO" + chr(10) + "    return ~ok",
    ),
    (
        "10 float32 sum-check tolerance huge",
        "src/hyperchaos/search3d.py",
        "F32_SUM_TOL = 0.05",
        "F32_SUM_TOL = 1e9",
    ),
    (
        "11 float32 lambda_1 threshold hugely negative",
        "src/hyperchaos/search3d.py",
        "F32_EPS_L1 = 0.005",
        "F32_EPS_L1 = -1e9",
    ),
    (
        "12 float32 zero-exponent limit huge",
        "src/hyperchaos/search3d.py",
        "F32_ZERO = 0.05",
        "F32_ZERO = 1e9",
    ),
    (
        "13 orbit transform drops the monomial sign factor",
        "src/hyperchaos/orbits.py",
        "                    sign *= s[inv[j]]  # x_j = s_inv[j] * y_inv[j]",
        "                    pass",
    ),
    (
        "14 orbit transform leaves the equations in place (renames variables only)",
        "src/hyperchaos/orbits.py",
        "new_e = inv[e]  # equation e",
        "new_e = e  # equation e",
    ),
]

FAST = ["tests/test_h2_search.py", "tests/test_orbits.py", "-k", "g5"]


def run_tests():
    env = dict(os.environ, PYTHONPATH="src", H1_T="5000")
    p = subprocess.run(
        [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x", *FAST],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return p.returncode, p.stdout


def main():
    status = subprocess.run(["git", "status", "--porcelain", "src", "tests"], cwd=ROOT, capture_output=True, text=True)
    if status.stdout.strip():
        print("tree not clean, commit first:\n" + status.stdout)
        sys.exit(2)
    rc0, out0 = run_tests()
    print("baseline exit", rc0, out0.strip().splitlines()[-1])
    if rc0 != 0:
        print("baseline is red; a red suite makes every bug look caught. Aborting.")
        sys.exit(3)
    for name, path, old, new in BUGS:
        full = os.path.join(ROOT, path)
        src = open(full, encoding="utf-8").read()
        if old not in src:
            print("BUG TEXT NOT FOUND:", name)
            continue
        open(full, "w", encoding="utf-8", newline="\n").write(src.replace(old, new, 1))
        try:
            rc, out = run_tests()
        finally:
            subprocess.run(["git", "checkout", "--", path], cwd=ROOT, check=True)
        failed = [l for l in out.splitlines() if l.startswith("FAILED")]
        print("bug", name, "-> exit", rc, "CAUGHT" if rc != 0 else "NOT CAUGHT", failed[:2])
    status = subprocess.run(["git", "status", "--porcelain", "src", "tests"], cwd=ROOT, capture_output=True, text=True)
    print("tree clean after restore:", not status.stdout.strip())


if __name__ == "__main__":
    main()
