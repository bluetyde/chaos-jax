"""Control systems for H1. Equations are copied from LITERATURE.md (sections A, C, D) or are textbook forms.

`expected` is the label the plan requires. `published` is the published spectrum, where one exists.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class System:
    id: str
    f: object
    x0: tuple
    width: float
    dt: float = 0.01
    expected: str = ""
    published: tuple = ()
    note: str = ""

    @property
    def n(self):
        return len(self.x0)

    def start(self, seed):
        rng = np.random.default_rng(seed)
        return np.asarray(self.x0) + self.width * rng.uniform(-1.0, 1.0, self.n)


def lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    def f(x):
        return jnp.stack([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]])

    return f


def rossler76(a=0.2, b=0.2, c=5.7):
    def f(x):
        return jnp.stack([-x[1] - x[2], x[0] + a * x[1], b + x[2] * (x[0] - c)])

    return f


def rossler79(a=0.25, b=3.0, c=0.5, d=0.05):
    def f(x):
        return jnp.stack([-(x[1] + x[2]), x[0] + a * x[1] + x[3], b + x[0] * x[2], -c * x[2] + d * x[3]])

    return f


def li_sprott(a=2.6, b=0.44):
    def f(x):
        return jnp.stack([x[1] - x[0], -x[0] * x[2] + x[3], x[0] * x[1] - a, -b * x[1]])

    return f


def cs13(A=3.6):
    def f(x):
        return jnp.stack([x[1], x[2], x[3], -(x[0] ** 4) * x[3] - A * x[2] - x[1] - x[0]])

    return f


def cs12():
    def f(x):
        return jnp.stack(
            [x[1], x[2], x[3], -0.1 * x[3] + (3 * x[0] - 4 * x[2]) * x[2] ** 2 - x[1] ** 3 - 2 * x[0] ** 3]
        )

    return f


def kopp25(a=0.62, b=0.55):
    def f(x):
        return jnp.stack([a * (-x[0] + x[1]) + x[3], -x[2] * jnp.tanh(x[0]), jnp.abs(x[0]) - 1.0, -b * x[0]])

    return f


def _sprott(case):
    forms = {
        "A": lambda x, y, z: (y, -x + y * z, 1 - y**2),
        "B": lambda x, y, z: (y * z, x - y, 1 - x * y),
        "C": lambda x, y, z: (y * z, x - y, 1 - x**2),
        "F": lambda x, y, z: (y + z, -x + 0.5 * y, x**2 - z),
        "S": lambda x, y, z: (-x - 4 * y, x + z**2, 1 + x),
    }
    g = forms[case]

    def f(v):
        return jnp.stack(g(v[0], v[1], v[2]))

    return f


def _hopf_pair(w1, w2, extra=None):
    def f(v):
        r1 = v[0] ** 2 + v[1] ** 2
        r2 = v[2] ** 2 + v[3] ** 2
        return jnp.stack(
            [
                v[0] * (1 - r1) - w1 * v[1],
                v[1] * (1 - r1) + w1 * v[0],
                v[2] * (1 - r2) - w2 * v[3],
                v[3] * (1 - r2) + w2 * v[2],
            ]
        )

    return f


def _hopf_plus_decay():
    def f(v):
        r = v[0] ** 2 + v[1] ** 2
        return jnp.stack([v[0] * (1 - r) - v[1], v[1] * (1 - r) + v[0], -v[2], -2.0 * v[3]])

    return f


def _hopf2d():
    def f(v):
        r = v[0] ** 2 + v[1] ** 2
        return jnp.stack([v[0] * (1 - r) - v[1], v[1] * (1 - r) + v[0]])

    return f


def _vdp(mu=1.0):
    def f(v):
        return jnp.stack([v[1], mu * (1 - v[0] ** 2) * v[1] - v[0]])

    return f


def _damped():
    def f(v):
        return jnp.stack([v[1], -v[0] - 0.5 * v[1]])

    return f


def _grow():
    def f(v):
        return jnp.stack([v[0]])

    return f


POSITIVE = [
    System("L", lorenz(), (1.0, 1.0, 1.0), 5.0, expected="chaotic", published=(0.906, 0.0, -14.573)),
    System("R76", rossler76(), (1.0, 1.0, 1.0), 1.0, expected="chaotic"),
    System("SB", _sprott("B"), (0.05, 0.05, 0.05), 0.05, expected="chaotic", published=(0.210, 0.0, -1.210)),
    System("SC", _sprott("C"), (0.05, 0.05, 0.05), 0.05, expected="chaotic", published=(0.163, 0.0, -1.163)),
    System("SF", _sprott("F"), (0.05, 0.05, 0.05), 0.05, expected="chaotic", published=(0.117, 0.0, -0.617)),
    System("SS", _sprott("S"), (0.05, 0.05, 0.05), 0.05, expected="chaotic", published=(0.188, 0.0, -1.188)),
    System("R79", rossler79(), (-10.0, -6.0, 0.0, 10.0), 0.2, dt=0.002, expected="hyperchaotic"),
    System(
        "LS", li_sprott(), (1.0, 1.0, 1.0, 1.0), 0.5, expected="hyperchaotic", published=(0.0704, 0.0128, 0.0, -1.0832)
    ),
    System(
        "CS13", cs13(), (0.1, 0.1, 0.1, 0.1), 0.05, expected="hyperchaotic", published=(0.132, 0.035, 0.0, -1.25)
    ),
    System(
        "CS12", cs12(), (0.1, 0.1, 0.1, 0.1), 0.05, expected="hyperchaotic", published=(0.141, 0.0168, 0.0, -0.257)
    ),
    System("KP", kopp25(), (1.0, 1.0, 1.0, 1.0), 0.3, expected="hyperchaotic"),
]

NEGATIVE = [
    System("N-fix", _damped(), (1.0, 0.0), 0.5, expected="fixed point"),
    System("N-div", _grow(), (1.0,), 0.5, expected="diverges"),
    System("N-lor14", lorenz(rho=14.0), (1.0, 1.0, 1.0), 5.0, expected="fixed point"),
    System("N-per", _hopf2d(), (0.5, 0.0), 0.3, expected="periodic"),
    System("N-vdp", _vdp(), (1.0, 0.0), 0.5, expected="periodic"),
    System("N-per4", _hopf_plus_decay(), (0.5, 0.0, 1.0, 1.0), 0.3, expected="periodic"),
    System("N-qp", _hopf_pair(1.0, float(np.sqrt(2.0))), (0.5, 0.0, 0.5, 0.0), 0.3, expected="quasi-periodic"),
    System("N-lor160", lorenz(rho=160.0), (1.0, 1.0, 1.0), 5.0, expected="periodic", note="from memory"),
    System("N-cons", _sprott("A"), (0.05, 0.05, 0.05), 0.05, expected="chaotic", note="conservative flag expected"),
]

ALL = {s.id: s for s in POSITIVE + NEGATIVE}
