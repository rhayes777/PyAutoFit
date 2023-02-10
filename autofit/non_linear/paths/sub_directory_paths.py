from abc import ABC
from typing import cast

from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.paths.database import DatabasePaths
from autofit.non_linear.paths.directory import DirectoryPaths


class SubDirectoryPaths(ABC):
    analysis_name: str
    parent: AbstractPaths

    def __new__(
            cls,
            parent=None,
            analysis_name=None,
            is_flat=False,
    ):
        """
        Determine which kind of sub-directory paths to generate
        """
        if isinstance(
                parent,
                DatabasePaths
        ):
            return super(SubDirectoryPaths, cls).__new__(SubDirectoryPathsDatabase)
        return super(SubDirectoryPaths, cls).__new__(SubDirectoryPathsDirectory)

    @property
    def _output_path(self) -> str:
        """
        The output path is customised to place output in a named directory in
        the analyses directory.
        """
        return f"{self.parent.output_path}/{self.analysis_name}"


class SubDirectoryPathsDirectory(
    DirectoryPaths,
    SubDirectoryPaths
):
    def __init__(
            self,
            parent: DirectoryPaths,
            analysis_name: str,
            is_flat=False,
    ):
        """
        Paths for a child directory for a search which generates multiple searches
        such as in sensitivity mapping or graphical modelling

        Parameters
        ----------
        parent
            A paths object for the original search
        analysis_name
            The name of the analysis (used to generate a directory name)
        is_flat
            If true then any children generated retain the top level parent
            rather than recursively taking a sub-directory parent.
        """
        self.analysis_name = analysis_name
        super().__init__()
        if is_flat and isinstance(
                parent,
                SubDirectoryPaths
        ):
            self.parent = parent.parent
        else:
            self.parent = parent

    @property
    def output_path(self) -> str:
        """
        The output path is customised to place output in a named directory in
        the analyses directory.
        """
        return self._output_path


class SubDirectoryPathsDatabase(
    DatabasePaths,
    SubDirectoryPaths
):
    def __init__(
            self,
            parent: DatabasePaths,
            analysis_name: str,
            is_flat=False,
    ):
        """
        Paths for a child directory for a search which generates multiple searches
        such as in sensitivity mapping or graphical modelling. Uses database to save
        most output data.

        Parameters
        ----------
        parent
            A paths object for the original search
        analysis_name
            The name of the analysis (used to generate a directory name)
        is_flat
            If true then any children generated retain the top level parent
            rather than recursively taking a sub-directory parent.
        """
        self.analysis_name = analysis_name
        super().__init__(
            parent.session
        )
        if is_flat and isinstance(
                parent,
                SubDirectoryPaths
        ):
            self.parent = cast(
                DatabasePaths,
                parent.parent
            )
        else:
            self.parent = parent

    @property
    def output_path(self) -> str:
        """
        The output path is customised to place output in a named directory in
        the analyses directory.
        """
        return self._output_path
