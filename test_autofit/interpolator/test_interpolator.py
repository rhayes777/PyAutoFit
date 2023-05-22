import pytest

import autofit as af
from autoconf.dictable import as_dict
from autofit.interpolator.covariance import LinearAnalysis, LinearRelationship
import numpy as np


def test_trivial():
    instance = af.ModelInstance(dict(t=1))
    linear_interpolator = af.LinearInterpolator([instance])

    result = linear_interpolator[linear_interpolator.t == 1]

    assert result is instance


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
                dict(
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
        model=af.Model(
            af.Gaussian,
            centre=0.0,
            normalization=1.0,
            sigma=-1.0,
        )
    )

    instance_1 = af.Collection(
        t=1.0,
        collection=collection,
    ).instance_from_prior_medians()
    instance_2 = af.Collection(
        t=2.0,
        collection=collection,
    ).instance_from_prior_medians()

    linear_interpolator = af.LinearInterpolator([instance_1, instance_2])

    result = linear_interpolator[linear_interpolator.t == 1.5]

    assert result.collection.model.centre == 0.0
    assert result.collection.model.normalization == 1.0
    assert result.collection.model.sigma == -1.0


def test_to_dict(linear_interpolator, linear_interpolator_dict):
    assert linear_interpolator.dict() == linear_interpolator_dict


def test_from_dict(linear_interpolator_dict):
    interpolator = af.LinearInterpolator.from_dict(linear_interpolator_dict)
    assert interpolator[interpolator.t == 1.5].t == 1.5


@pytest.fixture(name="instance_dict")
def make_instance_dict():
    return {
        "child_items": {
            "gaussian": {
                "centre": 0.0,
                "normalization": 1.0,
                "sigma": -1.0,
                "type": "autofit.example.model.Gaussian",
            },
            "t": 1.0,
            "type": "dict",
        },
        "type": "autofit.mapper.model.ModelInstance",
    }


@pytest.fixture(name="linear_interpolator_dict")
def make_linear_interpolator_dict(instance_dict):
    return {
        "instances": [
            instance_dict,
            {
                "child_items": {
                    "gaussian": {
                        "centre": 1.0,
                        "normalization": 2.0,
                        "sigma": -2.0,
                        "type": "autofit.example.model.Gaussian",
                    },
                    "t": 2.0,
                    "type": "dict",
                },
                "type": "autofit.mapper.model.ModelInstance",
            },
        ],
        "type": "autofit.interpolator.linear.LinearInterpolator",
    }


def test_instance_as_dict(model_instance, instance_dict):
    assert as_dict(model_instance) == instance_dict


def test_instance_from_dict(model_instance, instance_dict):
    instance = af.ModelInstance.from_dict(instance_dict)
    assert instance.t == 1.0

    gaussian = instance.gaussian
    assert isinstance(gaussian, af.Gaussian)
    assert gaussian.centre == 0.0
    assert gaussian.normalization == 1.0
    assert gaussian.sigma == -1.0


def test_trivial_linear_analysis():
    linear_analysis = LinearAnalysis(
        x=[1.0, 2.0, 3.0],
        y=[2.0, 4.0, 6.0],
        inverse_covariance_matrix=np.eye(3),
    )
    instance = 3 * [
        LinearRelationship(
            m=2.0,
            c=0.0,
        )
    ]
    assert list(linear_analysis._y(instance)) == [2.0, 4.0, 6.0]
    assert linear_analysis.log_likelihood_function(instance) == 0.0


# Multiple attributes per point in time
