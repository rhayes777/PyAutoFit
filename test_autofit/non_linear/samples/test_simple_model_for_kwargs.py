from autofit.non_linear.samples.util import simple_model_for_kwargs


def test_path_regression():
    kwargs = {
        ("galaxies", "lens", "shear", "angle"): 1.0,
        ("galaxies", "lens", "shear", "magnitude"): 2.0,
    }
    model = simple_model_for_kwargs(kwargs)

    shear = model.galaxies.lens.shear
    assert shear.angle.mean == 1.0
    assert shear.magnitude.mean == 2.0
