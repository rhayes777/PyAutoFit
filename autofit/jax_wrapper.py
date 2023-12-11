"""
Allows the user to switch between using NumPy and JAX for linear algebra operations.

If USE_JAX=1 then JAX's NumPy is used, otherwise vanilla NumPy is used.
"""
from os import environ

use_jax = environ.get("USE_JAX", "0") == "1"

import numpy  # noqa


def jit(function, *_, **__):
    return function


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
