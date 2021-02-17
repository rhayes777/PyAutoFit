import os
import shutil
from importlib import reload
from os import path

import pytest

import autofit as af

link_dir = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")

temp_folder_path = path.join(link_dir, "linked_folder")


@pytest.fixture(
    autouse=True
)
def make_files_directory():
    try:
        os.mkdir(link_dir)
    except FileExistsError:
        pass
    yield
    shutil.rmtree(link_dir, ignore_errors=True)


def delete_trees(*paths):
    for path in paths:
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)


class TestCase:
    def test_create_dir(self):
        assert path.exists(af.link.autolens_dir)

    def test_environment_model(self):
        symdir = path.join("~", ".symdir")
        os.environ["SYMDIR"] = symdir
        reload(af.link)
        assert ".symdir" in af.link.autolens_dir
        assert path.exists(af.link.autolens_dir)

    def test_consistent_dir(self):
        directory = af.link.path_for(path.join("a", "random", "directory"))
        assert af.link.autolens_dir in directory
        assert directory == af.link.path_for(path.join("a", "random", "directory"))
        assert directory != af.link.path_for(path.join("b", "random", "directory"))
        assert af.link.path_for(path.join("linked_file")) != af.link.path_for(
            path.join("linked_folder")
        )

        link_1 = af.link.path_for(
            path.join(
                "",
                "abcdefghijklmnopqrstuv",
                "a",
                "long",
                "directory",
                "that",
                "has",
                "a",
                "difference",
                "in",
                "the",
                "middle",
                "of",
                "the",
                "path" "",
                "abcdefghijklmnopqrstuv",
            )
        )

        link_2 = af.link.path_for(
            path.join(
                "",
                "abcdefghijklmnopqrstuv",
                "a",
                "long",
                "directory",
                "that",
                "is",
                "different",
                "in",
                "the",
                "middle",
                "of",
                "the",
                "path" "",
                "abcdefghijklmnopqrstuv",
            )
        )

        assert link_1 != link_2

    def test_make_linked_folder(self):

        if path.exists(temp_folder_path):
            os.remove(temp_folder_path)

        linked_path = af.link.make_linked_folder(temp_folder_path)
        assert af.link.autolens_dir in linked_path
        assert path.exists(linked_path)
        assert path.exists(temp_folder_path)
        delete_trees(linked_path, temp_folder_path)

    def test_longer_path(self):
        long_folder_path = path.join("tmp", "folder", "path")
        with pytest.raises(FileNotFoundError):
            af.link.make_linked_folder(long_folder_path)

    def test_clean_source(self):
        linked_path = af.link.make_linked_folder(temp_folder_path)
        temp_file_path = path.join("{}".format(linked_path), "{}".format("temp"))
        open(temp_file_path, "a").close()
        assert path.exists(temp_file_path)

        delete_trees(temp_folder_path)
        assert not path.exists(temp_folder_path)
        af.link.make_linked_folder(temp_folder_path)
        assert not path.exists(temp_file_path)

    def test_not_clean_source(self):
        linked_path = af.link.make_linked_folder(temp_folder_path)
        temp_file_path = path.join("{}".format(linked_path), "{}".format("temp"))
        open(temp_file_path, "a").close()
        assert path.exists(temp_file_path)

        af.link.make_linked_folder(temp_folder_path)
        assert path.exists(temp_file_path)
