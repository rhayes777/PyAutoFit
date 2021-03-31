import autofit as af
from autofit.mock.mock import Gaussian


def test_model_to_json():
    model = af.Model(
        Gaussian
    )

    assert model.dict == {
        "class_path": "autofit.mock.mock.Gaussian",
        "centre": {'lower_limit': 0.0, 'type': 'Uniform', 'upper_limit': 1.0},
        "intensity": {'lower_limit': 0.0, 'type': 'Uniform', 'upper_limit': 1.0},
        "sigma": {'lower_limit': 0.0, 'type': 'Uniform', 'upper_limit': 1.0},
    }
