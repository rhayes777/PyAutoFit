import autofit as af


def test_from_dict():
    model_dict = {
        "class_path": "autofit.example.model.Gaussian",
        "type": "model",
        "arguments": {
            "centre": {
                "lower_limit": 0.0,
                "upper_limit": 1.0,
                "type": "Uniform",
                "id": 2,
            },
            "normalization": {
                "lower_limit": 0.0,
                "upper_limit": 1.0,
                "type": "Uniform",
                "id": 1,
            },
            "sigma": {
                "lower_limit": 0.0,
                "upper_limit": 1.0,
                "type": "Uniform",
                "id": 2,
            },
        },
    }
    model = af.Model.from_dict(model_dict)
    assert model.prior_count == 2


def test_from_model():
    model = af.Model(af.Gaussian)

    model.centre = model.sigma

    assert model.prior_count == 2

    model_dict = model.dict()

    new_model = af.Model.from_dict(model_dict)

    assert new_model.prior_count == 2
    assert new_model.centre is new_model.sigma


def test_collection():
    collection = af.Collection(
        gaussian_1=af.Model(af.Gaussian),
        gaussian_2=af.Model(af.Gaussian),
    )
    collection.gaussian_1.centre = collection.gaussian_2.centre

    collection_dict = collection.dict()

    new_collection = af.Collection.from_dict(collection_dict)

    assert new_collection.gaussian_1.centre is new_collection.gaussian_2.centre
    assert new_collection.gaussian_1.sigma is not new_collection.gaussian_2.sigma
