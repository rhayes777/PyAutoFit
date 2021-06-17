import os
from pathlib import Path
from typing import List

from .file import File
from .item import Item


class Package(Item):
    def __init__(
            self,
            path: Path,
            packages: List["Package"],
            files: List[File],
            prefix: str,
            is_top_level: bool
    ):
        """
        A package in the project.

        Parameters
        ----------
        path
            The path to the package before edenisation
        packages
            Direct child packages
        files
            Direct child files
        prefix
            A prefix that must be prepended to all packages and modules
        is_top_level
            Is this the top level package of the project?
        """
        super().__init__(prefix)
        self._path = path
        self.packages = packages
        self.files = files
        self.is_top_level = is_top_level

        for child in self.children:
            child.parent = self

    @property
    def children(self) -> List[Item]:
        """
        Packages and files contained directly in this package
        """
        return [
            *self.packages,
            *self.files
        ]

    @property
    def path(self) -> Path:
        """
        The path to this package prior to edenisation
        """
        return self._path

    @classmethod
    def from_directory(
            cls,
            directory: Path,
            prefix: str,
            is_top_level=True,
    ):
        """
        Create a package object for a given directory

        Parameters
        ----------
        directory
            The directory of the package
        prefix
            A prefix that must be added to all packages and modules
        is_top_level
            Is this the top level package in the project?

        Returns
        -------
        A representation of the package
        """
        files = list()
        children = list()

        for item in os.listdir(
                directory
        ):
            path = directory / item
            if item.endswith(".py"):
                files.append(
                    File(path, prefix)
                )

            if os.path.isdir(path):
                if "__init__.py" in os.listdir(
                        path
                ):
                    children.append(
                        Package.from_directory(
                            path,
                            prefix=prefix,
                            is_top_level=False
                        )
                    )
        return Package(
            directory,
            children,
            files,
            prefix,
            is_top_level
        )
