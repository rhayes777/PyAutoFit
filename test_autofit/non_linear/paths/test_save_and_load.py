import autofit as af


def test_directory_json():
    paths = af.DirectoryPaths()
    dictionary = {"hello": "world"}
    paths.save_json("test", dictionary)
    assert paths.load_json("test") == dictionary
