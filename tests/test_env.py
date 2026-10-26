import jax.numpy as jnp

import chaos_jax  # noqa: F401  (importing turns on x64)


def test_float64_is_on():
    assert jnp.zeros(1).dtype == jnp.float64
    assert jnp.asarray(1.0).dtype == jnp.float64
