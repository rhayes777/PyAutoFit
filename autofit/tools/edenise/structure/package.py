import os
from pathlib import Path
from typing import List, Optional

from .file import File
from .item import Item


class Package(Item):
    def __init__(
            self,
            path: Path,
            prefix: str,
            is_top_level: bool,
            parent: Optional["Package"] = None
    ):
        """
        A package in the project.

        Parameters
        ----------
        path
            The path to the package before edenisation
        prefix
            A prefix that must be prepended to all packages and modules
        is_top_level
            Is this the top level package of the project?
        """
        super().__init__(
            prefix,
            parent=parent
        )
        self._path = path
        self._children = list()

        for item in os.listdir(
                path
        ):
            item_path = path / item
            if item.endswith(".py"):
                self.children.append(
                    File(
                        item_path,
                        prefix,
                        parent=self
                    )
                )

            if os.path.isdir(item_path):
                if "__init__.py" in os.listdir(
                        item_path
                ):
                    self.children.append(
                        Package(
                            item_path,
                            prefix=prefix,
                            is_top_level=False,
                            parent=self
                        )
                    )

        self.is_top_level = is_top_level

    @property
    def children(self) -> List[Item]:
        """
        Packages and files contained directly in this package
        """
        return self._children

    @property
    def path(self) -> Path:
        """
        The path to this package prior to edenisation
        """
        return self._path
