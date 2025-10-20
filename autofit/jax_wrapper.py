import logging

logger = logging.getLogger(__name__)

"""
Allows the user to switch between using NumPy and JAX for linear algebra operations.

If USE_JAX=true in general.yaml then JAX's NumPy is used, otherwise vanilla NumPy is used.
"""
from autoconf import conf

use_jax = conf.instance["general"]["jax"]["use_jax"]

if use_jax:

    import os

    xla_env = os.environ.get("XLA_FLAGS")

    xla_env_set = True

    if xla_env is None:
        xla_env_set = False
    elif isinstance(xla_env, str):
        xla_env_set = "--xla_disable_hlo_passes=constant_folding" in xla_env

    if not xla_env_set:
        logger.info(
            """
            For fast JAX compile times, the envirment variable XLA_FLAGS must be set to "--xla_disable_hlo_passes=constant_folding",
            which is currently not.
            
            In Python, to do this manually, use the code: 
            
            import os
            os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=constant_folding"
            
            The environment variable has been set automatically for you now, however if JAX has already been imported, 
            this change will not take effect and JAX function compiling times may be slow. 
            
            Therefore, it is recommended to set this environment variable before running your script, e.g. in your terminal.
            """)

        os.environ['XLA_FLAGS'] = "--xla_disable_hlo_passes=constant_folding"

    import jax
    from jax import numpy

    print(

    """
***JAX ENABLED*** 
    
Using JAX for grad/jit and GPU/TPU acceleration. 
To disable JAX, set: config -> general -> jax -> use_jax = false
    """)

    def jit(function, *args, **kwargs):
        return jax.jit(function, *args, **kwargs)

    def grad(function, *args, **kwargs):
        return jax.grad(function, *args, **kwargs)


else:

    print(
    """
***JAX DISABLED*** 
    
Falling back to standard NumPy (no grad/jit or GPU support).
To enable JAX (if supported), set: config -> general -> jax -> use_jax = true
    """)

    import numpy  # noqa

    def jit(function, *_, **__):
        return function

    def grad(function, *_, **__):
        return function

from jax._src.tree_util import (
    register_pytree_node_class as register_pytree_node_class,
    register_pytree_node as register_pytree_node,
)

