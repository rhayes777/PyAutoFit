import os
from os import path

import pytest

import autofit as af

directory = path.dirname(path.realpath(__file__))




class PatchPaths(af.Paths):
    @property
    def path(self):
        return f"{directory}/path"

    @property
    def backup_path(self) -> str:
        return f"{directory}/backup_path"

    @property
    def sym_path(self) -> str:
        return f"{directory}/sym_path"

    @property
    @af.make_path
    def output_path(self) -> str:
        return f"{directory}/phase_output_path"


@pytest.fixture(name="paths")
def make_paths():
    paths = PatchPaths(remove_files=True)
    return paths


def test_backup_zip_remove(paths):
    try:
        os.mkdir(paths.sym_path)
    except FileExistsError:
        pass

    try:
        os.mkdir(paths.path)
    except FileExistsError:
        pass

    paths.backup_zip_remove()

    assert not os.path.exists(paths.path)
    assert not os.path.exists(f"{directory}/phase_output_path")
    assert os.path.exists(paths.backup_path)
    assert os.path.exists(paths.zip_path)

    os.remove(paths.zip_path)
    os.rmdir(paths.sym_path)
    os.rmdir(paths.backup_path)


def test_restore(paths):
    os.mkdir(paths.sym_path)
    os.mkdir(paths.path)

    paths.backup_zip_remove()

    os.rmdir(paths.sym_path)

    paths.restore()

    assert os.path.exists(paths.output_path)
    assert os.path.exists(paths.backup_path)
    assert not os.path.exists(paths.zip_path)

    os.rmdir(paths.backup_path)
    os.rmdir(paths.output_path)
