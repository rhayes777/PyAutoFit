def test_jsons(search_output):
    assert len(list(search_output.names_and_paths(".json"))) == 3


def test_id(search_output):
    assert isinstance(search_output.id, str)
