import autofit as af


def test_reference(model_dict):
    model_dict.pop("class_path")
    with_path = {"gaussian": model_dict, "type": "collection"}
    reference = {"gaussian": "autofit.example.model.Gaussian"}
    model = af.AbstractModel.from_dict(
        with_path,
        reference=reference,
    )

    assert model.gaussian.cls is af.Gaussian
