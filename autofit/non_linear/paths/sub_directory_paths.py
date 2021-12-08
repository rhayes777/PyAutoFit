from abc import ABC

from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.paths.database import DatabasePaths
from autofit.non_linear.paths.directory import DirectoryPaths


class SubDirectoryPaths(AbstractPaths, ABC):
    def __init__(
            self,
            parent: AbstractPaths,
            analysis_name: str
    ):
        """
        Manages output paths for an analysis that is the child of another
        analysis, for example a single analysis in a combined analysis.

        Parameters
        ----------
        parent
            Paths for the parent analysis
        analysis_name
            A name for this analysis
        """
        self.analysis_name = analysis_name
        super().__init__()
        self.parent = parent

    def __new__(
            cls,
            parent,
            analysis_name,
    ):
        if isinstance(
                parent,
                DatabasePaths
        ):
            return object.__new__(SubDirectoryPathsDatabase)
        return object.__new__(SubDirectoryPathsDirectory)

    @property
    def output_path(self) -> str:
        """
        The output path is customised to place output in a named directory in
        the analyses directory.
        """
        return f"{self.parent.output_path}/{self.analysis_name}"


class SubDirectoryPathsDirectory(
    DirectoryPaths,
    SubDirectoryPaths
):
    @property
    def output_path(self) -> str:
        return super(
            SubDirectoryPaths,
            self,
        ).output_path


class SubDirectoryPathsDatabase(
    DatabasePaths,
    SubDirectoryPaths
):
    @property
    def output_path(self) -> str:
        return super(
            SubDirectoryPaths,
            self,
        ).output_path
