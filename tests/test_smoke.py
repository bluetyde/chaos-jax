import jax.numpy as jnp

import hyperchaos


def test_float64_is_on():
    assert hyperchaos.__version__ == "0.0.0"
    assert jnp.zeros(1).dtype == jnp.float64
