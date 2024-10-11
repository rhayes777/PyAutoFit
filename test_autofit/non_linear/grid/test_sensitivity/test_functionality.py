import pytest

import autofit as af
from autofit.non_linear.grid import sensitivity as s


def test_lists(sensitivity):
    assert len(list(sensitivity._perturb_instances)) == 8


def test_tuple_step_size(sensitivity):
    sensitivity.number_of_steps = (2, 2, 4)

    assert len(sensitivity._lists) == 16


def test_labels(sensitivity):
    labels = list(sensitivity._labels)
    assert labels == [
        "centre_0.25_normalization_0.25_sigma_0.25",
        "centre_0.25_normalization_0.25_sigma_0.75",
        "centre_0.25_normalization_0.75_sigma_0.25",
        "centre_0.25_normalization_0.75_sigma_0.75",
        "centre_0.75_normalization_0.25_sigma_0.25",
        "centre_0.75_normalization_0.25_sigma_0.75",
        "centre_0.75_normalization_0.75_sigma_0.25",
        "centre_0.75_normalization_0.75_sigma_0.75",
    ]


def test_perform_job(job):
    assert not job.is_complete

    result = job.perform()
    assert isinstance(result, s.JobResult)
    assert isinstance(result.perturb_result, af.Result)
    assert isinstance(result.result, af.Result)

    assert job.is_complete


def test_perform_twice(job):
    job.perform()
    assert job.is_complete

    result = job.perform()
    assert isinstance(result, s.JobResult)
    assert isinstance(result.perturb_result, af.Result)
    assert isinstance(result.result, af.Result)

    assert job.is_complete


class TestPerturbationModels:
    @pytest.mark.parametrize(
        "limit_scale, fl, fu, sl, su",
        [
            (
                1.0,
                0.0,
                0.5,
                0.5,
                1.0,
            ),
            (
                2.0,
                0.0,
                0.75,
                0.25,
                1.0,
            ),
            (
                4.0,
                0.0,
                1.0,
                0.0,
                1.0,
            ),
        ],
    )
    def test_perturb_models(self, sensitivity, limit_scale, fl, fu, sl, su):
        sensitivity.limit_scale = limit_scale
        jobs = sensitivity._make_jobs()
        models = [job.perturb_model for job in jobs]

        first, second, *_ = models

        assert first is not second

        assert first.sigma.lower_limit == fl
        assert first.sigma.upper_limit == fu
        assert second.sigma.lower_limit == sl
        assert second.sigma.upper_limit == su

    def test__perturb_models__prior_overwrite_via_perturb_model_prior_func(
        self,
        sensitivity,
    ):
        def perturb_model_prior_func(perturb_instance, perturb_model):
            perturb_model.centre = af.UniformPrior(lower_limit=-7.0, upper_limit=4.0)

            return perturb_model

        sensitivity.perturb_model_prior_func = perturb_model_prior_func
        jobs = sensitivity._make_jobs()
        models = [job.perturb_model for job in jobs]

        first, second, *_ = models

        assert first is not second

        assert first.centre.lower_limit == -7.0
        assert first.centre.upper_limit == 4.0

    def test_physical(self, sensitivity):
        sensitivity.perturb_model.sigma = af.UniformPrior(upper_limit=10)
        model = list(sensitivity._make_jobs())[0].perturb_model
        assert model.sigma.upper_limit == 5

    def test_model_with_limits(self):
        model = af.Model(af.Gaussian)

        with_limits = model.with_limits(
            [
                (0.3, 0.5),
                (0.3, 0.5),
                (0.3, 0.5),
            ]
        )
        assert with_limits.centre.lower_limit == 0.3
        assert with_limits.centre.upper_limit == 0.5

    def test_prior_with_limits(self):
        prior = af.UniformPrior(
            lower_limit=0,
            upper_limit=10,
        ).with_limits(3, 5)
        assert prior.lower_limit == 3
        assert prior.upper_limit == 5

    def test_existing_limits(self):
        prior = af.UniformPrior(2, 4).with_limits(3, 5)
        assert prior.lower_limit == 3
        assert prior.upper_limit == 4


@pytest.fixture(name="tuple_sensitivity")
def make_tuple_sensitivity(sensitivity):
    sensitivity.number_of_steps = (2, 2, 4)
    return sensitivity


def test_models_with_tuple_steps(tuple_sensitivity):
    tuple_sensitivity.number_of_steps = (2, 2, 4)
    assert len(list(tuple_sensitivity._perturb_models)) == 16


def test_jobs_with_tuple_steps(tuple_sensitivity):
    assert len(list(tuple_sensitivity._make_jobs())) == 16
