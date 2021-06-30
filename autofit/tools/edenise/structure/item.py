from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Optional


class Item(ABC):
    def __init__(
            self,
            prefix: str = "",
            parent: Optional["Item"] = None
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
        self._parent = parent

        if prefix or self._parent is None:
            self.prefix = prefix
        else:
            self.prefix = self._parent.prefix

    def __getitem__(self, item):
        for child in self.children:
            if child.name == item:
                return child
        member = Member(item)
        member.parent = self
        return member

    @property
    def target_file_name(self):
        return self.target_name

    @property
    def should_remove_type_annotations(self):
        return self.top_level.should_remove_type_annotations

    @property
    def should_rename_modules(self):
        return self.top_level.should_rename_modules

    @property
    def eden_dependencies(self):
        return self.top_level.eden_dependencies

    @property
    def target_import_string(self) -> str:
        return self.target_name

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
    def target_path(self) -> Path:
        """
        The path this object will have after edenisation
        """
        target_file_name = self.target_file_name
        if self.parent is None:
            return Path(target_file_name)
        return self.parent.target_path / target_file_name

    @property
    def target_import_path(self) -> str:
        """
        The path by which this object will be imported after edenisation
        """
        if self.parent is None:
            return self.target_name
        return f"{self.parent.target_import_path}.{self.target_name}"

    @property
    def import_path(self) -> str:
        if self.parent is None:
            return self.name
        return f"{self.parent.import_path}.{self.name}"


class DirectoryItem(Item, ABC):
    @abstractmethod
    def generate_target(self, output_path):
        pass

    @abstractmethod
    def target_file_name(self):
        pass


class Member(Item):
    def __init__(self, name):
        super().__init__()
        self._name = name

    @property
    def target_name(self) -> str:
        return self.name

    @property
    def path(self) -> Path:
        return self.parent.path / self._name

    @property
    def children(self) -> List["Item"]:
        return []
