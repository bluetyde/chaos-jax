"""chaos-jax: gradients through chaotic simulations."""
import jax

# Chaos experiments need float64: float32 rounding grows like exp(lambda*T) too.
jax.config.update("jax_enable_x64", True)
