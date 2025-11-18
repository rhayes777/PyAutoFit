import pickle

import autofit as af

from test_autofit.graphical.gaussian.model import Analysis, Gaussian, make_data
from test_autofit.graphical.gaussian import model as model_module

import pytest

# jax = pytest.importorskip("jax")
#
#
#
# @pytest.fixture(autouse=True, name="model")
# def make_model():
#     return af.Model(Gaussian)
#
#
# @pytest.fixture(name="analysis")
# def make_analysis():
#     import jax.numpy as jnp
#     x = jnp.arange(100)
#     y = make_data(Gaussian(centre=50.0, normalization=25.0, sigma=10.0), x)
#     return Analysis(x, y)


# @pytest.fixture(name="instance")
# def make_instance():
#     return Gaussian()
#
#
# def test_jit_likelihood(analysis, instance):
#
#     import jax
#
#     instance = Gaussian()
#
#     jitted = jax.jit(analysis.log_likelihood_function)
#
#     assert jitted(instance) == analysis.log_likelihood_function(instance)


# def test_jit_dynesty_static(
#     analysis,
#     model,
#     monkeypatch,
# ):
#     monkeypatch.setattr(
#         jax_wrapper,
#         "use_jax",
#         True,
#     )
#     search = af.DynestyStatic(
#         use_gradient=True,
#         number_of_cores=1,
#         maxcall=1,
#     )
#
#     print(search.fit(model=model, analysis=analysis))
#
#     loaded = pickle.loads(pickle.dumps(search))
#     assert isinstance(loaded, af.DynestyStatic)
