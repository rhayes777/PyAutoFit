from os import path
import pytest

from autoconf import conf
import autofit as af

from autofit.non_linear.samples import MCMCSamples
from autofit.mock.mock import MockClassx4

directory = path.dirname(path.realpath(__file__))
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="samples")
def make_samples():
    model = af.ModelMapper(mock_class_1=MockClassx4)

    parameters = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
        [0.0, 1.0, 2.0, 3.0],
    ]

    return MCMCSamples(
        model=model,
        parameters=parameters,
        log_likelihoods=[1.0, 2.0, 3.0, 10.0, 5.0],
        log_priors=[0.0, 0.0, 0.0, 0.0, 0.0],
        weights=[1.0, 1.0, 1.0, 1.0, 1.0],
        auto_correlation_times=1,
        auto_correlation_check_size=2,
        auto_correlation_required_length=3,
        auto_correlation_change_threshold=4,
        total_walkers=5,
        total_steps=6,
        time=7,
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance.push(
        path.join(directory, "files", "emcee", "config"),
        output_path=path.join(directory, "files", "emcee", "output"),
    )


class TestJsonCSV:
    def test__from_csv_table_and_json_info(self, samples):

        mcmc = af.Emcee()

        samples.write_table(filename=path.join(mcmc.paths.samples_path, "samples.csv"))
        samples.info_to_json(filename=path.join(mcmc.paths.samples_path, "info.json"))

        model = af.ModelMapper(mock_class_1=MockClassx4)

        samples = mcmc.samples_via_csv_json_from_model(model=model)

        assert samples.parameters == [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [21.0, 22.0, 23.0, 24.0],
            [0.0, 1.0, 2.0, 3.0],
        ]
        assert samples.log_likelihoods == [1.0, 2.0, 3.0, 10.0, 5.0]
        assert samples.log_priors == [0.0, 0.0, 0.0, 0.0, 0.0]
        assert samples.log_posteriors == [1.0, 2.0, 3.0, 10.0, 5.0]
        assert samples.weights == [1.0, 1.0, 1.0, 1.0, 1.0]
        #  assert samples.auto_correlation_times == None
        assert samples.auto_correlation_check_size == 2
        assert samples.auto_correlation_required_length == 3
        assert samples.auto_correlation_change_threshold == 4
        assert samples.total_walkers == 5
        assert samples.total_steps == 6
        assert samples.time == 7
