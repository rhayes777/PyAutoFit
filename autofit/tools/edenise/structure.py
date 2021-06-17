import os
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
        self.parent = None
        self.prefix = prefix

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

    @property
    def target_name(self) -> str:
        """
        The name this object will be given after Edenisation
        """
        suffix = "".join(
            string.title()
            for string
            in self.name.split("_")
        )
        return f"{self.prefix}_{suffix}"

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


class Import(Item):
    def __init__(self, string: str):
        """
        An import statement in a file

        Parameters
        ----------
        string
            The original line describing the import
        """
        super().__init__()
        self.string = string

        self.loc = {}
        exec(
            f"""
{self.string}
import inspect

is_class = inspect.isclass({self.suffix})

if is_class:
    path = inspect.getfile({self.suffix})
else:
    path = {self.suffix}.__file__
""",
            globals(),
            self.loc
        )

    @property
    def children(self):
        """
        Imports don't have any children
        """
        return []

    @property
    def file(self) -> "File":
        """
        The file containing this import
        """
        path = str(self.path)

        def get_from_item(
                item
        ):
            for child in item.children:
                if path == str(child.path):
                    return child
                if path.startswith(
                        str(child.path)
                ):
                    return get_from_item(
                        child
                    )

        return get_from_item(
            self.top_level
        )

    @property
    def target_import_string(self) -> str:
        """
        The string that will describe this import after edenisation
        """
        return f"from {self.file.target_import_path} import {self.suffix}"

    @property
    def suffix(self) -> str:
        """
        The final component of this import.

        e.g. from os import path -> path
        """
        return self.string.split(" ")[-1]

    # def is_class(self) -> bool:
    #     return self.loc["is_class"]

    @property
    def path(self) -> Path:
        """
        The path to the file containing this import
        """
        return Path(self.loc["path"])


class File(Item):
    def __init__(
            self,
            path: Path,
            prefix: str
    ):
        """
        A file to be edenised

        Parameters
        ----------
        path
            The path to the file prior to edenisation
        prefix
            A prefix to be prepended to package and module names
        """
        super().__init__(prefix)
        self._path = path

    @property
    def name(self) -> str:
        """
        The name of the file without the .py suffix
        """
        return super().name.replace(".py", "")

    @property
    def children(self) -> List[Import]:
        """
        Imports in the file
        """
        return self.imports

    @property
    def path(self) -> Path:
        """
        The path to the file
        """
        return self._path

    @property
    def imports(self) -> List[Import]:
        """
        Imports in the file
        """
        imports = list()
        with open(self.path) as f:
            for line in f.read().split("\n"):
                if line.startswith(
                        "from"
                ) or line.startswith(
                    "import"
                ):
                    import_ = Import(line)
                    import_.parent = self
                    imports.append(
                        import_
                    )
        return imports

    @property
    def project_imports(self) -> List[Import]:
        """
        Imports in the file that belong to the project
        """
        return [
            import_
            for import_
            in self.imports
            if import_.is_in_project
        ]


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
