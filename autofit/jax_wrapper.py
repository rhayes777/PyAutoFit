"""
Allows the user to switch between using NumPy and JAX for linear algebra operations.

If USE_JAX=1 then JAX's NumPy is used, otherwise vanilla NumPy is used.
"""
from os import environ

use_jax = environ.get("USE_JAX", "0") == "1"

if use_jax:
    try:
        import jax
        from jax import numpy

        def jit(function, *args, **kwargs):
            return jax.jit(function, *args, **kwargs)

        print("JAX mode enabled")
    except ImportError:
        raise ImportError(
            "JAX is not installed. Please install it with `pip install jax`."
        )
else:
    import numpy  # noqa
    from scipy.special.cython_special import erfinv  # noqa

    def jit(function, *_, **__):
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
