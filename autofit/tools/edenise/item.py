from abc import abstractmethod, ABC
from functools import lru_cache
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

    @lru_cache()
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
    def top_level(self):
        """
        The top level package in the project
        """
        if self.parent is None:
            return self
        return self.parent.top_level

    def first_ancestor_with_type(self, item_type):
        """
        The first ancestor with a given type.

        e.g. the file containing a line or import
        """
        if self.parent is None:
            raise AttributeError(
                f"No ancestor found with type {item_type}"
            )
        if isinstance(
                self.parent,
                item_type
        ):
            return self.parent
        return self.first_ancestor_with_type(
            item_type
        )

    @property
    def path(self) -> Path:
        """
        The path to this package or file
        """
        return self.parent.path

    @property
    def children(self) -> List["Item"]:
        """
        Packages, files or imports that are direct descendents
        of this item
        """
        return []

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


class DirectoryItem(Item, ABC):
    @abstractmethod
    def generate_target(self, output_path):
        pass

    @abstractmethod
    def target_file_name(self):
        pass

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"


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
