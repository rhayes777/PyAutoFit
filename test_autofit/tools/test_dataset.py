from test_autofit.mock import MockDataset


def test_save_and_load():
    dataset = MockDataset()
    dataset.save("/tmp")
    assert dataset.name == MockDataset.load(f"/tmp/{dataset.name}.pickle").name
