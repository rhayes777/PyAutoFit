"""
Allows the user to switch between using NumPy and JAX for linear algebra operations.

If USE_JAX=1 then JAX's NumPy is used, otherwise vanilla NumPy is used.
"""
from autoconf import conf

use_jax = conf.instance["general"]["jax"]["use_jax"]

if use_jax:
    try:
        import jax
        from jax import numpy

        def jit(function, *args, **kwargs):
            return jax.jit(function, *args, **kwargs)

        def grad(function, *args, **kwargs):
            return jax.grad(function, *args, **kwargs)

        print("JAX mode enabled")
    except ImportError:
        raise ImportError(
            """
            JAX is not installed, but the use_jax setting in config -> general.yaml is true. 
            
            Please install it with `pip install jax` or set the use_jax setting to false.
            """
        )
else:
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
