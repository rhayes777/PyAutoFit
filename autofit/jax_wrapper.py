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

        print("JAX mode enabled")
    except ImportError:
        raise ImportError(
            "JAX is not installed. Please install it with `pip install jax`."
        )
else:
    import numpy  # noqa
