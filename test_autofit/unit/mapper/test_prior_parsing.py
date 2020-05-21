import pytest

import autofit as af
from autofit.mapper.prior.deferred import DeferredArgument


@pytest.fixture(name="uniform_dict")
def make_uniform_dict():
    return {"type": "Uniform", "lower_limit": 2, "upper_limit": 3}


@pytest.fixture(name="uniform_prior")
def make_uniform_prior(uniform_dict):
    return af.Prior.from_dict(uniform_dict)


@pytest.fixture(name="log_uniform_dict")
def make_log_uniform_dict():
    return {"type": "LogUniform", "lower_limit": 0.2, "upper_limit": 0.3}


@pytest.fixture(name="log_uniform_prior")
def make_log_uniform_prior(log_uniform_dict):
    return af.Prior.from_dict(log_uniform_dict)


@pytest.fixture(name="gaussian_dict")
def make_gaussian_dict():
    return {
        "type": "Gaussian",
        "lower_limit": -10,
        "upper_limit": 10,
        "mean": 3,
        "sigma": 4,
    }


@pytest.fixture(name="gaussian_prior")
def make_gaussian_prior(gaussian_dict):
    return af.Prior.from_dict(gaussian_dict)


@pytest.fixture(name="relative_width_dict")
def make_relative_width_dict():
    return {"type": "Relative", "value": 1.0}


@pytest.fixture(name="absolute_width_dict")
def make_absolute_width_dict():
    return {"type": "Absolute", "value": 2.0}


@pytest.fixture(name="relative_width_modifier")
def make_relative_width_modifier(relative_width_dict):
    return af.prior.WidthModifier.from_dict(relative_width_dict)


@pytest.fixture(name="absolute_width_modifier")
def make_absolute_width_modifier(absolute_width_dict):
    return af.prior.WidthModifier.from_dict(absolute_width_dict)


class TestWidth:
    def test_relative(self, relative_width_modifier):
        assert isinstance(relative_width_modifier, af.prior.RelativeWidthModifier)
        assert relative_width_modifier.value == 1.0

    def test_absolute(self, absolute_width_modifier):
        assert isinstance(absolute_width_modifier, af.prior.AbsoluteWidthModifier)
        assert absolute_width_modifier.value == 2.0


class TestDict:
    def test_uniform(self, uniform_prior, uniform_dict):
        assert uniform_prior.dict == uniform_dict

    def test_log_uniform(self, log_uniform_prior, log_uniform_dict):
        assert log_uniform_prior.dict == log_uniform_dict

    def test_gaussian(self, gaussian_prior, gaussian_dict):
        assert gaussian_prior.dict == gaussian_dict


class TestFromDict:
    def test_uniform(self, uniform_prior):
        assert isinstance(uniform_prior, af.UniformPrior)
        assert uniform_prior.lower_limit == 2
        assert uniform_prior.upper_limit == 3

    def test_log_uniform(self, log_uniform_prior, absolute_width_modifier):
        assert isinstance(log_uniform_prior, af.LogUniformPrior)
        assert log_uniform_prior.lower_limit == 0.2
        assert log_uniform_prior.upper_limit == 0.3

    def test_gaussian(self, gaussian_prior):
        assert isinstance(gaussian_prior, af.GaussianPrior)
        assert gaussian_prior.lower_limit == -10
        assert gaussian_prior.upper_limit == 10
        assert gaussian_prior.mean == 3
        assert gaussian_prior.sigma == 4

    def test_constant(self):
        result = af.Prior.from_dict({"type": "Constant", "value": 1.5})
        assert result == 1.5

    def test_deferred(self):
        result = af.Prior.from_dict({"type": "Deferred"})
        assert isinstance(result, DeferredArgument)
