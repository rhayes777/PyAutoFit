from os import path

import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__samples_via_csv_from():

    search = af.PySwarmsGlobal()
    search.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "pyswarms"))
    search.paths._identifier = "tag"

    model = af.ModelMapper(mock_class=af.m.MockClassx4)
    model.mock_class.two = af.LogUniformPrior(lower_limit=1e-8, upper_limit=10.0)

    samples = search.samples_via_csv_from(model=model)

    assert isinstance(samples.parameter_lists, list)
    assert isinstance(samples.parameter_lists[0], list)
    assert isinstance(samples.log_likelihood_list, list)
    assert isinstance(samples.log_prior_list, list)
    assert isinstance(samples.log_posterior_list, list)
    assert isinstance(samples.weight_list, list)

    assert samples.parameter_lists[0] == pytest.approx(
        [0.0, 2.0, 3.0, 1.0], 1.0e-4
    )
    assert samples.log_likelihood_list[0] == pytest.approx(1.0, 1.0e-4)
    assert samples.log_prior_list[0] == pytest.approx(0.0, 1.0e-4)
    assert samples.weight_list[0] == pytest.approx(1.0, 1.0e-4)
