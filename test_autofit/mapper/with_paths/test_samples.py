import pytest

import autofit as af


@pytest.fixture(
    name="sample"
)
def make_sample():
    return af.Sample(
        log_likelihood=0,
        log_prior=0,
        weight=0,
        kwargs={
            ("gaussian_1", "centre",): 1,
            ("gaussian_1", "normalization",): 2,
            ("gaussian_1", "sigma",): 3,
            ("gaussian_2", "centre",): 4,
            ("gaussian_2", "normalization",): 5,
            ("gaussian_2", "sigma",): 6,
        }
    )


@pytest.fixture(
    name="samples"
)
def make_samples(
        model,
        sample
):
    return af.Samples(
        model=model,
        sample_list=[sample],
    )


class TestWith:
    def test_trivial(
            self,
            sample
    ):
        with_paths = sample.with_paths([
            ("gaussian_1", "centre",)
        ])

        assert with_paths.kwargs == {
            ("gaussian_1", "centre",): 1
        }

    def test_subpath(
            self,
            sample
    ):
        with_paths = sample.with_paths([
            ("gaussian_1",)
        ])

        assert with_paths.kwargs == {
            ("gaussian_1", "centre",): 1,
            ("gaussian_1", "normalization",): 2,
            ("gaussian_1", "sigma",): 3,
        }

    def test_samples(
            self,
            samples
    ):
        with_paths = samples.with_paths([
            ("gaussian_1",)
        ])

        assert with_paths.sample_list[0].kwargs == {
            ("gaussian_1", "centre",): 1,
            ("gaussian_1", "normalization",): 2,
            ("gaussian_1", "sigma",): 3,
        }

        model = with_paths.model
        assert hasattr(
            model,
            "gaussian_1"
        )
        assert not hasattr(
            model,
            "gaussian_2"
        )


class TestWithout:
    def test_trivial(
            self,
            sample
    ):
        without_paths = sample.without_paths([
            ("gaussian_1", "centre",)
        ])

        assert without_paths.kwargs == {
            ("gaussian_1", "normalization",): 2,
            ("gaussian_1", "sigma",): 3,
            ("gaussian_2", "centre",): 4,
            ("gaussian_2", "normalization",): 5,
            ("gaussian_2", "sigma",): 6,
        }

    def test_subpath(
            self,
            sample
    ):
        without_paths = sample.without_paths([
            ("gaussian_1",)
        ])

        assert without_paths.kwargs == {
            ("gaussian_2", "centre",): 4,
            ("gaussian_2", "normalization",): 5,
            ("gaussian_2", "sigma",): 6,
        }

    def test_samples(
            self,
            samples,
    ):
        without_paths = samples.without_paths([
            ("gaussian_1",)
        ])

        assert without_paths.sample_list[0].kwargs == {
            ("gaussian_2", "centre",): 4,
            ("gaussian_2", "normalization",): 5,
            ("gaussian_2", "sigma",): 6,
        }

        model = without_paths.model
        assert not hasattr(
            model,
            "gaussian_1"
        )
        assert hasattr(
            model,
            "gaussian_2"
        )


def test_samples_lazy_attributes(
        samples
):
    paths = samples.paths
    names = samples.names

    without_paths = samples.without_paths([])

    assert without_paths.paths is not paths
    assert without_paths.names is not names


def test_tuples(samples):
    model = af.Collection(
        lens=af.Model(
            af.m.MockWithTuple
        )
    )
    samples = af.Samples(
        model=model,
        sample_list=[
            af.Sample(
                log_likelihood=0,
                log_prior=0,
                weight=0,
                kwargs={
                    path: i
                    for i, path
                    in enumerate(
                        model.paths
                    )
                }
            )
        ]
    )
    samples = samples.without_paths(
        [
            ("lens", "tup",),
        ]
    )
    assert len(samples.parameter_lists) == 1
