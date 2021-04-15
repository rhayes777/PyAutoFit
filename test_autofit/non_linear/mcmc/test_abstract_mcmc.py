from os import path

import pytest

import autofit as af
from autofit.mock.mock import MockClassx4
from autofit.non_linear.samples import MCMCSamples, Sample
from autofit.non_linear.mcmc.auto_correlations import AutoCorrelations

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="samples")
def make_samples():
    model = af.ModelMapper(mock_class_1=MockClassx4)
    print(model.path_priors_tuples)

    parameters = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
        [0.0, 1.0, 2.0, 3.0],
    ]

    samples = [
        Sample(
            log_likelihood=1.0,
            log_prior=0.0,
            weights=1.0,
            mock_class_1_one=0.0,
            mock_class_1_two=1.0,
            mock_class_1_three=2.0,
            mock_class_1_four=3.0
        ),
        Sample(
            log_likelihood=2.0,
            log_prior=0.0,
            weights=1.0,
            mock_class_1_one=0.0,
            mock_class_1_two=1.0,
            mock_class_1_three=2.0,
            mock_class_1_four=3.0
        ),
        Sample(
            log_likelihood=3.0,
            log_prior=0.0,
            weights=1.0,
            mock_class_1_one=0.0,
            mock_class_1_two=1.0,
            mock_class_1_three=2.0,
            mock_class_1_four=3.0
        ),
        Sample(
            log_likelihood=10.0,
            log_prior=0.0,
            weights=1.0,
            mock_class_1_one=21.0,
            mock_class_1_two=22.0,
            mock_class_1_three=23.0,
            mock_class_1_four=24.0
        ),
        Sample(
            log_likelihood=5.0,
            log_prior=0.0,
            weights=1.0,
            mock_class_1_one=0.0,
            mock_class_1_two=1.0,
            mock_class_1_three=2.0,
            mock_class_1_four=3.0
        )
    ]

    return MCMCSamples(
        model=model,
        samples=samples,
        auto_correlations=AutoCorrelations(
            times=1,
            check_size=2,
            required_length=3,
            change_threshold=4,
            previous_times=5
        ),
        total_walkers=5,
        total_steps=6,
        time=7,
    )


class TestJsonCSV:
    def test__from_csv_table_and_json_info(self, samples):
        mcmc = af.Emcee()
        mcmc.paths = af.DirectoryPaths(path_prefix=path.join("non_linear", "emcee"))

        mcmc.paths._identifier = "tag"

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
        assert samples.auto_correlations.times == pytest.approx([31.98507049, 36.51001152, 73.4762926, 67.67495736], 1.0e-4)
        assert samples.auto_correlations.check_size == 2
        assert samples.auto_correlations.required_length == 3
        assert samples.auto_correlations.change_threshold == 4
        assert samples.total_walkers == 5
        assert samples.total_steps == 6
        assert samples.auto_correlations.times[-1] == pytest.approx(67.67, 1.0e-4)
