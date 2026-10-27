import os
from functools import lru_cache

import pytest

import hyperchaos  # noqa: F401  (turns on float64 before anything else)
from hyperchaos.classify import classify
from hyperchaos.spectrum import spectrum
from hyperchaos.systems import ALL
from hyperchaos.verify import verify

SEED = 1
T_MAIN = float(os.environ.get("H1_T", "5000"))


@lru_cache(maxsize=None)
def main_run(sid):
    s = ALL[sid]
    return spectrum(s.f, s.start(SEED), dt=s.dt, T=T_MAIN)


@lru_cache(maxsize=None)
def main_class(sid):
    return classify(main_run(sid))


@lru_cache(maxsize=None)
def verification(sid):
    s = ALL[sid]
    return verify(s, main_run(sid), s.start(SEED), T=T_MAIN, v4_T=0.4 * T_MAIN)


@pytest.fixture(scope="session")
def run():
    return main_run


@pytest.fixture(scope="session")
def cls():
    return main_class


@pytest.fixture(scope="session")
def ver():
    return verification
