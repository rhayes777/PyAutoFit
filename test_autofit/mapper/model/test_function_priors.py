import autofit as af
import pytest
from autonerves import conf
from autonerves.exc import ConfigException


def configured_function(value=0.0):
    return value


def test_function_config_prior(tmp_path):
    priors = tmp_path / "priors"
    priors.mkdir()
    (priors / "test_function_priors.yaml").write_text(
        "configured_function:\n  value:\n    type: Uniform\n    lower_limit: 2.0\n    upper_limit: 4.0\n"
    )
    original_configs = conf.instance.configs
    conf.instance.push(new_path=tmp_path)
    try:
        model = af.Model(configured_function)
        assert model.cls is configured_function
        assert model.value.lower_limit == 2.0
        assert model.value.upper_limit == 4.0
        assert model.instance_from_prior_medians() == 3.0
    finally:
        conf.instance.configs = original_configs


@pytest.mark.parametrize("frozen", [False, True])
def test_function_explicit_prior(frozen):
    prior = af.UniformPrior(lower_limit=2.0, upper_limit=4.0)
    model = af.Model(configured_function, value=prior)
    assert model.value is prior
    if frozen:
        model.freeze()
    assert model.instance_from_prior_medians() == 3.0


def test_function_missing_config_has_actionable_error():
    with pytest.raises(
        ConfigException, match="configured_function.*value.*explicit prior"
    ):
        af.Model(configured_function)


def test_class_initializer_keeps_owner_config():
    model = af.Model(af.ex.Gaussian.__init__)
    result = model.instance_from_prior_medians()
    assert isinstance(result, af.ex.Gaussian)
    assert model.prior_count == 3
