from abc import abstractmethod, ABC
from pathlib import Path
from typing import List


class Item(ABC):
    def __init__(
            self,
            prefix: str = ""
    ):
        """
        A package, file or import in a project to be converted
        to Eden

        Parameters
        ----------
        prefix
            Some prefix to be added to the names of directories and files
            for the hell of it apparently
        """
        self._parent = None
        self.prefix = prefix

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent
        self.prefix = parent.prefix

    @property
    def top_level(self) -> "Item":
        """
        The top level package in the project
        """
        if self.parent is None:
            return self
        return self.parent.top_level

    @property
    @abstractmethod
    def path(self) -> Path:
        """
        The path to this package or file
        """

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    @property
    @abstractmethod
    def children(self) -> List["Item"]:
        """
        Packages, files or imports that are direct descendents
        of this item
        """

    @property
    def is_in_project(self) -> bool:
        """
        Is this object within the top level object?
        """
        return str(self.path).startswith(
            str(self.top_level.path)
        )

    @property
    def name(self) -> str:
        """
        The current name of this object
        """
        return self.path.name

    def _edenise_string(self, string):
        suffix = "".join(
            string.title()
            for string
            in string.split("_")
        )
        return f"{self.prefix}_{suffix}"

    @property
    def target_name(self) -> str:
        """
        The name this object will be given after Edenisation
        """
        return self._edenise_string(
            self.name
        )

    @property
    def target_path(self) -> str:
        """
        The path this object will have after edenisation
        """
        target_name = self.target_name
        if self.parent is None:
            return Path(target_name)
        return self.parent.target_path / target_name

    @property
    def target_import_path(self) -> str:
        """
        The path by which this object will be imported after edenisation
        """
        if self.parent is None:
            return self.target_name
        return f"{self.parent.target_import_path}.{self.target_name}"
