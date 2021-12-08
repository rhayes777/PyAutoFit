from pathlib import Path

import autofit as af
from autofit.non_linear.paths.sub_directory_paths import (
    SubDirectoryPaths, SubDirectoryPathsDirectory,
    SubDirectoryPathsDatabase
)


def test_directory():
    paths = af.DirectoryPaths()
    subdirectory_path = SubDirectoryPaths(
        parent=paths,
        analysis_name="name"
    )
    assert isinstance(
        subdirectory_path,
        SubDirectoryPathsDirectory
    )
    assert subdirectory_path.output_path == str(Path(
        paths.output_path
    ) / "name")


def test_database(session):
    paths = af.DatabasePaths(session)
    subdirectory_path = SubDirectoryPaths(
        parent=paths,
        analysis_name="name"
    )
    assert isinstance(
        subdirectory_path,
        SubDirectoryPathsDatabase
    )
    assert subdirectory_path.output_path == str(Path(
        paths.output_path
    ) / "name")


def test_is_flat():
    paths = af.DirectoryPaths()
    subdirectory_path = SubDirectoryPaths(
        parent=paths,
        analysis_name="name",
        is_flat=True
    )
    assert subdirectory_path.parent is paths

    subdirectory_path = SubDirectoryPaths(
        parent=subdirectory_path,
        analysis_name="name",
        is_flat=True,
    )
    assert subdirectory_path.parent is paths
