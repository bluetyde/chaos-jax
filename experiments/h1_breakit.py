"""Planted-bug check for H1 (gate G6). Run only on a clean, committed tree.

Each bug is written into the source, the fast tests are run, and the file is restored with git.
Usage: python experiments/h1_breakit.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

BUGS = [
    (
        "1 drop the QR re-orthonormalisation (columns only normalised)",
        "src/hyperchaos/spectrum.py",
        "        q, r = jnp.linalg.qr(q)\n        ssum = ssum + jnp.log(jnp.abs(jnp.diag(r)))\n",
        "        nrm = jnp.linalg.norm(q, axis=0)\n        q = q / nrm\n        ssum = ssum + jnp.log(nrm)\n",
    ),
    (
        "2 transposed Jacobian in the tangent equation",
        "src/hyperchaos/spectrum.py",
        "        return f(x), j @ q, jnp.trace(j)\n",
        "        return f(x), j.T @ q, jnp.trace(j)\n",
    ),
    (
        "3 classifier ignores eps (positive means above 0)",
        "src/hyperchaos/classify.py",
        "    pos = lam > eps_i\n",
        "    pos = lam > 0.0\n",
    ),
    (
        "4 classifier skips the sum check (Amendment 4)",
        "src/hyperchaos/classify.py",
        "    if abs(float(np.sum(res.exponents)) - res.mean_div) > SUM_TOL_REL",
        "    if False and abs(float(np.sum(res.exponents)) - res.mean_div) > SUM_TOL_REL",
    ),
]

FAST = ["tests/test_h1_controls.py", "tests/test_h1_classifier.py", "-k", "(g1 or g2 or g3 or g4) and not KP and not CS12 and not CS13"]


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
