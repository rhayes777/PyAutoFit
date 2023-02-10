import os
import shutil
from os import path

import pytest

import autofit as af

directory = path.dirname(path.realpath(__file__))


class PatchPaths(af.DirectoryPaths):
    @property
    def sym_path(self) -> str:
        return path.join(directory, "sym_path")

    @property
    def output_path(self) -> str:
        return path.join(directory, "phase_output_path")


@pytest.fixture(name="paths")
def make_paths():
    return PatchPaths()


def test_restore(paths):
    paths.model = af.Model(af.Gaussian)
    paths.save_all({}, {}, [])

    paths.zip_remove()
    paths.restore()

    assert path.exists(paths.output_path)
    assert not path.exists(paths._zip_path)

    shutil.rmtree(paths.output_path)
