import pytest

import autofit as af

@pytest.fixture(
    name="model"
)
def make_model():
    return af.Collection(
        gaussian=af.Model(
            af.Gaussian
        )
    )


class TestInstanceFromPathArguments:
    def test(
            self,
            model
    ):
        instance = model.instance_from_path_arguments({
            ("gaussian", "centre"): 0.1,
            ("gaussian", "normalization"): 0.2,
            ("gaussian", "sigma"): 0.3
        })
        assert instance.gaussian.centre == 0.1
        assert instance.gaussian.normalization == 0.2
        assert instance.gaussian.sigma == 0.3

    def test_prior_linking(
            self,
            model
    ):
        model.gaussian.centre = model.gaussian.normalization
        instance = model.instance_from_path_arguments({
            ("gaussian", "centre",): 0.1,
            ("gaussian", "sigma",): 0.3
        })
        assert instance.gaussian.centre == 0.1
        assert instance.gaussian.normalization == 0.1
        assert instance.gaussian.sigma == 0.3

        instance = model.instance_from_path_arguments({
            ("gaussian", "normalization",): 0.2,
            ("gaussian", "sigma",): 0.3
        })
        assert instance.gaussian.centre == 0.2
        assert instance.gaussian.normalization == 0.2
        assert instance.gaussian.sigma == 0.3


@pytest.fixture(
    name="underscore_model"
)
def make_underscore_model():
    return af.Collection(
        gaussian_component=af.Model(
            af.Gaussian
        )
    )


class TestInstanceFromPriorNames:
    def test(self, model):
        instance = model.instance_from_prior_name_arguments({
            "gaussian_centre": 0.1,
            "gaussian_normalization": 0.2,
            "gaussian_sigma": 0.3
        })
        assert instance.gaussian.centre == 0.1
        assert instance.gaussian.normalization == 0.2
        assert instance.gaussian.sigma == 0.3

    def test_underscored_names(self, underscore_model):
        instance = underscore_model.instance_from_prior_name_arguments({
            "gaussian_component_centre": 0.1,
            "gaussian_component_normalization": 0.2,
            "gaussian_component_sigma": 0.3
        })
        assert instance.gaussian_component.centre == 0.1
        assert instance.gaussian_component.normalization == 0.2
        assert instance.gaussian_component.sigma == 0.3

    def test_prior_linking(self, underscore_model):
        underscore_model.gaussian_component.normalization = (
            underscore_model.gaussian_component.centre
        )
        instance = underscore_model.instance_from_prior_name_arguments({
            "gaussian_component_centre": 0.1,
            "gaussian_component_sigma": 0.3
        })
        assert instance.gaussian_component.centre == 0.1
        assert instance.gaussian_component.normalization == 0.1
        assert instance.gaussian_component.sigma == 0.3

        instance = underscore_model.instance_from_prior_name_arguments({
            "gaussian_component_normalization": 0.2,
            "gaussian_component_sigma": 0.3
        })
        assert instance.gaussian_component.centre == 0.2
        assert instance.gaussian_component.normalization == 0.2
        assert instance.gaussian_component.sigma == 0.3

    def test_path_for_name(self, underscore_model):
        assert underscore_model.path_for_name(
            "gaussian_component_centre"
        ) == (
                   "gaussian_component",
                   "centre"
               )


def test_component_names():
    model = af.Model(
        af.Gaussian
    )
    assert model.model_component_and_parameter_names == [
        'centre', 'normalization', 'sigma'
    ]


def test_with_tuple():
    with_tuple = af.Model(
        af.m.MockWithTuple
    )
    assert with_tuple.model_component_and_parameter_names == [
        "tup_0", "tup_1"
    ]


@pytest.fixture(
    name="linked_model"
)
def make_linked_model():
    model = af.Model(
        af.Gaussian
    )
    model.sigma = model.centre
    return model


class TestAllPaths:
    def test_independent(self):
        model = af.Model(
            af.Gaussian
        )

        assert model.all_paths_prior_tuples == [
            ((("centre",),), model.centre),
            ((("normalization",),), model.normalization),
            ((("sigma",),), model.sigma),
        ]

    def test_linked(self, linked_model):
        assert linked_model.all_paths_prior_tuples == [
            ((("centre",), ("sigma",)), linked_model.centre),
            ((("normalization",),), linked_model.normalization)
        ]

    def test_names_independent(self):
        model = af.Model(
            af.Gaussian
        )

        assert model.all_name_prior_tuples == [
            (("centre",), model.centre),
            (("normalization",), model.normalization),
            (("sigma",), model.sigma),
        ]

    def test_names_linked(self, linked_model):
        assert linked_model.all_name_prior_tuples == [
            (("centre", "sigma"), linked_model.centre),
            (("normalization",), linked_model.normalization)
        ]


@pytest.fixture(
    name="samples"
)
def make_samples(model):
    return af.Samples(
        model,
        [
            af.Sample(
                log_likelihood=1.0,
                log_prior=1.0,
                weight=1.0,
                kwargs={
                    ("gaussian", "centre"): 0.1,
                    ("gaussian", "normalization"): 0.2,
                    ("gaussian", "sigma"): 0.3,
                }
            )
        ]
    )


@pytest.mark.parametrize(
    "path, value",
    [
        (("gaussian", "centre"), [0.1]),
        (("gaussian", "normalization"), [0.2]),
        (("gaussian", "sigma"), [0.3]),
    ]
)
def test_values_for_path(
        samples,
        path,
        value
):
    assert samples.values_for_path(
        path
    ) == value


@pytest.fixture(
    name="result"
)
def make_result(
        model,
        samples
):
    return af.Result(
        samples,
        model,
        af.m.MockSearch()
    )


@pytest.fixture(
    name="modified_result"
)
def make_modified_result(
        model,
        samples
):
    model.gaussian.sigma = af.GaussianPrior(
        mean=0.5,
        sigma=1
    )
    model.gaussian.centre = af.GaussianPrior(
        mean=0.5,
        sigma=1
    )
    return af.Result(
        samples,
        model,
        af.m.MockSearch()
    )


class TestFromResult:
    def test_instance(
            self,
            result
    ):
        instance = result.max_log_likelihood_instance

        assert instance.gaussian.centre == 0.1
        assert instance.gaussian.normalization == 0.2
        assert instance.gaussian.sigma == 0.3

    def test_model(
            self,
            result
    ):
        model = result.model

        assert model.gaussian.centre.mean == 0.1
        assert model.gaussian.normalization.mean == 0.2
        assert model.gaussian.sigma.mean == 0.3

    def test_modified_instance(
            self,
            modified_result
    ):
        instance = modified_result.max_log_likelihood_instance

        assert instance.gaussian.centre == 0.1
        assert instance.gaussian.normalization == 0.2
        assert instance.gaussian.sigma == 0.3

    def test_modified_model(
            self,
            modified_result
    ):
        model = modified_result.model

        assert model.gaussian.centre.mean == 0.1
        assert model.gaussian.normalization.mean == 0.2
        assert model.gaussian.sigma.mean == 0.3
