from abc import ABC

from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.paths.database import DatabasePaths
from autofit.non_linear.paths.directory import DirectoryPaths


class SubDirectoryPaths(ABC):
    analysis_name: str
    parent: AbstractPaths

    def __new__(
            cls,
            parent,
            analysis_name,
    ):
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
            analysis_name,
            parent,
    ):
        self.analysis_name = analysis_name
        super().__init__()
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
            analysis_name,
            parent: DatabasePaths,
    ):
        self.analysis_name = analysis_name
        super().__init__(
            parent.session
        )
        self.parent = parent

    @property
    def output_path(self) -> str:
        """
        The output path is customised to place output in a named directory in
        the analyses directory.
        """
        return self._output_path
