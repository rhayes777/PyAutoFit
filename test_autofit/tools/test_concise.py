import autofit as af


def test_model_info():
    collection = af.Collection([af.Model(af.Gaussian) for _ in range(20)])
    print(collection.info)
