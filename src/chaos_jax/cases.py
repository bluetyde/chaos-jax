"""Fixed choices for Phase 1. Rationale in docs/PHASE1-CHOICES.md."""
import jax.numpy as jnp

LORENZ_DT = 0.01
LORENZ_SPINUP_T = 50.0
LORENZ_LAMBDA_T = 2000.0
LORENZ_STARTS = (
    (1.0, 1.0, 1.0),
    (-5.0, 7.0, 30.0),
)

STEPS_PER_BLOCK = 10
DISCARD_BLOCKS = 100

PENDULUM_HIGH = (2.5, 2.5, 0.0, 0.0)
PENDULUM_LOW = (0.1, 0.1, 0.0, 0.0)
PENDULUM_LAMBDA_T = 1000.0
PENDULUM_ENERGY_T = 10.0
PENDULUM_DT = 0.00125  # from experiments/pendulum_dt_scan.py, see docs/PHASE1-CHOICES.md
PENDULUM_DT_LADDER = (0.02, 0.01, 0.005, 0.0025, 0.00125)
ENERGY_DRIFT_MARGIN = 10.0  # choose dt so drift is this many times below the gate

OSCILLATOR_START = (1.0, 0.0)


def vec(t):
    return jnp.array(t, dtype=jnp.float64)
