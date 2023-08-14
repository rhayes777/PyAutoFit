import os
import shutil
from os import path

import pytest

import autofit as af
from pathlib import Path

directory = Path(__file__).parent


class PatchPaths(af.DirectoryPaths):
    @property
    def sym_path(self) -> Path:
        return directory / "sym_path"

    @property
    def output_path(self) -> Path:
        return directory / "phase_output_path"


@pytest.fixture(name="paths")
def make_paths():
    return PatchPaths()


def test_restore(paths):
    paths.model = af.Model(af.Gaussian)
    paths.save_all({}, {})

    paths.zip_remove()
    paths.restore()

    assert path.exists(paths.output_path)
    assert not path.exists(paths._zip_path)

    shutil.rmtree(paths.output_path)
