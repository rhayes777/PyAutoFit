from os import environ

use_jax = environ.get("USE_JAX", "0") == "1"

if use_jax:
    try:
        import jax
    except ImportError:
        raise ImportError(
            "JAX is not installed. Please install it with `pip install jax`."
        )
