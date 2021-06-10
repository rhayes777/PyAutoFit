import os
from abc import abstractmethod, ABC
from pathlib import Path
from typing import List


class Item(ABC):
    def __init__(self, prefix=""):
        self.parent = None
        self.prefix = prefix

    @property
    def top_level(self):
        if self.parent is None:
            return self
        return self.parent.top_level

    @property
    @abstractmethod
    def path(self) -> Path:
        pass

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    @property
    @abstractmethod
    def children(self):
        pass

    @property
    def is_in_project(self):
        return str(self.path).startswith(
            str(self.top_level.path)
        )

    @property
    def name(self):
        return self.path.name

    @property
    def target_name(self):
        suffix = "".join(
            string.title()
            for string
            in self.name.split("_")
        )
        return f"{self.prefix}_{suffix}"

    @property
    def target_path(self):
        target_name = self.target_name
        if self.parent is None:
            return Path(target_name)
        return self.parent.target_path / target_name

    @property
    def target_import_path(self):
        if self.parent is None:
            return self.target_name
        return f"{self.parent.target_import_path}.{self.target_name}"


class Import(Item):
    def __init__(self, string):
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
        return []

    @property
    def target(self):
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
    def suffix(self):
        return self.string.split(" ")[-1]

    def is_class(self):
        return self.loc["path"]

    @property
    def path(self):
        return Path(self.loc["path"])


class File(Item):
    def __init__(self, path: Path, prefix):
        super().__init__(prefix)
        self._path = path

    @property
    def children(self):
        return self.imports

    @property
    def path(self):
        return self._path

    @property
    def imports(self):
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
    def project_imports(self):
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
        super().__init__(prefix)
        self._path = path
        self.packages = packages
        self.files = files
        self.is_top_level = is_top_level

        for child in self.children:
            child.parent = self

    @property
    def children(self):
        return self.packages + self.files

    @property
    def path(self):
        return self._path

    @classmethod
    def from_directory(
            cls,
            directory,
            prefix,
            is_top_level=True,
    ):
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
