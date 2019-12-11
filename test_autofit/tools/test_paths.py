import os

import pytest

import autofit as af


class PatchPaths(af.Paths):
    @property
    def path(self):
        return "path"

    @property
    def backup_path(self) -> str:
        return "backup_path"

    @property
    def sym_path(self) -> str:
        return "sym_path"


@pytest.fixture(
    name="paths"
)
def make_paths():
    paths = PatchPaths()
    os.mkdir(paths.sym_path)
    os.mkdir(paths.path)
    yield paths
    os.rmdir(paths.sym_path)


def test_backup_zip_remove(paths):
    paths.backup_zip_remove()

    assert not os.path.exists(paths.path)
    assert not os.path.exists(paths.backup_path)
    assert os.path.exists(paths.zip_path)

    os.remove(paths.zip_path)
