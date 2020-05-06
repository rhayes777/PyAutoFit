import os
import shutil
from importlib import reload

import pytest

import autofit as af

temp_folder_path = "/tmp/linked_folder"


def delete_trees(*paths):
    for path in paths:
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)


class TestCase:
    def test_create_dir(self):
        assert os.path.exists(af.link.autolens_dir)

    def test_environment_model(self):
        symdir = "~/.symdir"
        os.environ["SYMDIR"] = symdir
        reload(af.link)
        assert ".symdir" in af.link.autolens_dir
        assert os.path.exists(af.link.autolens_dir)

    def test_consistent_dir(self):
        directory = af.link.path_for("/a/random/directory")
        assert af.link.autolens_dir in directory
        assert directory == af.link.path_for("/a/random/directory")
        assert directory != af.link.path_for("/b/random/directory")
        assert af.link.path_for("/tmp/linked_file") != af.link.path_for(
            "/tmp/linked_folder"
        )

        link_1 = af.link.path_for(
            "/abcdefghijklmnopqrstuv/a/long/directory/that/has/a/difference/in/the/middle/of/the/path"
            "/abcdefghijklmnopqrstuv"
        )

        link_2 = af.link.path_for(
            "/abcdefghijklmnopqrstuv/a/long/directory/that/is/different/in/the/middle/of/the/path"
            "/abcdefghijklmnopqrstuv"
        )

        assert link_1 != link_2

    def test_make_linked_folder(self):
        path = af.link.make_linked_folder(temp_folder_path)
        assert af.link.autolens_dir in path
        assert os.path.exists(path)
        assert os.path.exists(temp_folder_path)
        delete_trees(path, temp_folder_path)

    def test_longer_path(self):
        long_folder_path = "/tmp/folder/path"
        with pytest.raises(FileNotFoundError):
            af.link.make_linked_folder(long_folder_path)

    def test_clean_source(self):
        path = af.link.make_linked_folder(temp_folder_path)
        temp_file_path = "{}/{}".format(path, "temp")
        open(temp_file_path, "a").close()
        assert os.path.exists(temp_file_path)

        delete_trees(temp_folder_path)
        assert not os.path.exists(temp_folder_path)
        af.link.make_linked_folder(temp_folder_path)
        assert not os.path.exists(temp_file_path)

    def test_not_clean_source(self):
        path = af.link.make_linked_folder(temp_folder_path)
        temp_file_path = "{}/{}".format(path, "temp")
        open(temp_file_path, "a").close()
        assert os.path.exists(temp_file_path)

        af.link.make_linked_folder(temp_folder_path)
        assert os.path.exists(temp_file_path)
