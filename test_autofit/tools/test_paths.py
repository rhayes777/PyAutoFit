import os
from os import path

import pytest

import autofit as af

directory = path.dirname(path.realpath(__file__))


class PatchPaths(af.DirectoryPaths):
    @property
    def path(self):
        return path.join(directory, "path")

    @property
    def sym_path(self) -> str:
        return path.join(directory, "sym_path")

    @property
    @af.make_path
    def output_path(self) -> str:
        return path.join(directory, "phase_output_path")


@pytest.fixture(name="paths")
def make_paths():
    paths = PatchPaths()
    return paths


def test_zip_remove(paths):
    try:
        os.mkdir(paths.sym_path)
    except FileExistsError:
        pass

    try:
        os.mkdir(paths.path)
    except FileExistsError:
        pass

    paths.zip_remove()

    assert not path.exists(paths.path)
    assert not path.exists(path.join(directory, "phase_output_path"))
    assert path.exists(paths._zip_path)

    os.remove(paths._zip_path)
    os.rmdir(paths.sym_path)


def test_restore(paths):
    os.mkdir(paths.sym_path)
    os.mkdir(paths.path)

    paths.zip_remove()

    os.rmdir(paths.sym_path)

    paths.restore()

    assert path.exists(paths.output_path)
    assert not path.exists(paths._zip_path)

    os.rmdir(paths.output_path)
