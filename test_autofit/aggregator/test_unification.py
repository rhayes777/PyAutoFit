import autofit as af


def test_id(search_output):
    assert isinstance(search_output.id, str)


def test_model(search_output):
    assert isinstance(search_output.model, af.AbstractModel)
