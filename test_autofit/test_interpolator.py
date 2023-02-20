import pytest

import autofit as af


def test_trivial():
    instance = af.ModelInstance(items=dict(t=1))
    linear_interpolator = af.LinearInterpolator([instance])

    result = linear_interpolator[linear_interpolator.t == 1]

    assert result is instance


@pytest.fixture(name="instances")
def make_instances():
    return [
        af.ModelInstance(
            items=dict(
                t=1.0, gaussian=af.Gaussian(centre=0.0, normalization=1.0, sigma=-1.0),
            )
        ),
        af.ModelInstance(
            items=dict(
                t=2.0, gaussian=af.Gaussian(centre=1.0, normalization=2.0, sigma=-2.0),
            )
        ),
    ]


@pytest.fixture(name="linear_interpolator")
def make_linear_interpolator(instances):
    return af.LinearInterpolator(instances)


def test_spline_interpolator(instances):
    interpolator = af.SplineInterpolator(instances)

    result = interpolator[interpolator.t == 1.5]

    assert result.t == 1.5
    assert result.gaussian.centre == 0.5


def test_smooth_spline_interpolator(instances):
    interpolator = af.SplineInterpolator(
        instances
        + [
            af.ModelInstance(
                items=dict(
                    t=3.0,
                    gaussian=af.Gaussian(centre=4.0, normalization=3.0, sigma=-3.0),
                )
            ),
        ]
    )

    result = interpolator[interpolator.t == 1.5]

    assert result.t == 1.5
    assert result.gaussian.centre < 0.5


@pytest.mark.parametrize(
    "t, centre", [(0.0, -1.0), (1.0, 0.0), (1.5, 0.5), (2.0, 1.0), (3.0, 2.0)]
)
def test_linear(t, centre, linear_interpolator):

    result = linear_interpolator[linear_interpolator.t == t]

    assert result.t == t
    assert result.gaussian.centre == centre
    assert result.gaussian.normalization == t
    assert result.gaussian.sigma == -t


@pytest.mark.parametrize("sigma", [-0.5, 0.0, 0.5, 1.0])
def test_alternate_attribute(linear_interpolator, sigma):

    result = linear_interpolator[linear_interpolator.gaussian.sigma == sigma]

    assert result.gaussian.sigma == sigma
    assert result.t == -sigma
    assert result.gaussian.normalization == -sigma


def test_deeper_attributes():
    collection = af.Collection(
        model=af.Model(af.Gaussian, centre=0.0, normalization=1.0, sigma=-1.0,)
    )

    instance_1 = af.Collection(
        t=1.0, collection=collection,
    ).instance_from_prior_medians()
    instance_2 = af.Collection(
        t=2.0, collection=collection,
    ).instance_from_prior_medians()

    linear_interpolator = af.LinearInterpolator([instance_1, instance_2])

    result = linear_interpolator[linear_interpolator.t == 1.5]

    assert result.collection.model.centre == 0.0
    assert result.collection.model.normalization == 1.0
    assert result.collection.model.sigma == -1.0


def test_to_dict(linear_interpolator):
    assert linear_interpolator.dict() == {}
