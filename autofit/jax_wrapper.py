"""
Allows the user to switch between using NumPy and JAX for linear algebra operations.

If USE_JAX=true in general.yaml then JAX's NumPy is used, otherwise vanilla NumPy is used.
"""
import jax

from autoconf import conf

use_jax = conf.instance["general"]["jax"]["use_jax"]

if use_jax:

    print("""
    JAX is enabled. Using JAX for grad/jit and GPU/TPU acceleration.
    To disable JAX, set: config -> general -> jax -> use_jax = false
    """)

    def jit(function, *args, **kwargs):
        return jax.jit(function, *args, **kwargs)

    def grad(function, *args, **kwargs):
        return jax.grad(function, *args, **kwargs)

else:

    print("""
    JAX is disabled. Falling back to standard NumPy (no grad/jit or GPU support).
    To enable JAX (if supported), set: config -> general -> jax -> use_jax = true
    """)

    import numpy  # noqa
    from scipy.special.cython_special import erfinv  # noqa

    def jit(function, *_, **__):
        return function

    def grad(function, *_, **__):
        return function

try:
    from jax._src.tree_util import (
        register_pytree_node_class as register_pytree_node_class,
        register_pytree_node as register_pytree_node,
    )
    from jax._src.scipy.special import erfinv

except ImportError:

    def register_pytree_node_class(cls):
        return cls

    def register_pytree_node(*_, **__):
        def decorator(cls):
            return cls

        return decorator
