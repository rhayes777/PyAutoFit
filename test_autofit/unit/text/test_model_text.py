import autofit as af
from test_autofit import mock
from test_autofit.mock import MockClassx4


def test__parameter_names_from_text():
    model = af.ModelMapper()
    model.ls = af.CollectionPriorModel(
        [
            af.PriorModel(mock.MockClassRelativeWidth),
            af.PriorModel(mock.MockClassRelativeWidth),
        ]
    )

    parameter_names = model.parameter_names

    assert parameter_names == [
        "ls_0_one",
        "ls_0_two",
        "ls_0_three",
        "ls_1_one",
        "ls_1_two",
        "ls_1_three",
    ]


def test__parameter_labels_from_text():
    model = af.PriorModel(MockClassx4)

    parameter_labels = af.text.Model.parameter_labels_from_model(model=model)

    assert parameter_labels == [
        r"x4p0_{\mathrm{a}}",
        r"x4p1_{\mathrm{a}}",
        r"x4p2_{\mathrm{a}}",
        r"x4p3_{\mathrm{a}}",
    ]
