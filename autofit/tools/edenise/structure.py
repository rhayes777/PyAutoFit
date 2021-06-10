import os
from abc import abstractmethod, ABC
from pathlib import Path
from typing import List


class Item(ABC):
    def __init__(self):
        self.parent = None

    @property
    def top_level(self):
        if self.parent is None:
            return self
        return self.parent.top_level

    @property
    @abstractmethod
    def path(self) -> Path:
        pass

    @property
    def is_in_project(self):
        return str(self.path).startswith(
            str(self.top_level.path)
        )


class Import(Item):
    def __init__(self, string):
        super().__init__()
        self.string = string

    @property
    def suffix(self):
        return self.string.split(" ")[-1]

    @property
    def path(self):
        loc = {}
        print(self.suffix)
        exec(
            f"""
{self.string}
import inspect

if inspect.isclass({self.suffix}):
    path = inspect.getfile({self.suffix})
else:
    path = {self.suffix}.__file__
""",
            globals(),
            loc
        )
        return Path(loc["path"])


class File(Item):
    def __init__(self, path: Path):
        super().__init__()
        self._path = path

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
            children: List["Package"],
            files: List[File],
            prefix: str,
            is_top_level: bool
    ):
        super().__init__()
        self._path = path
        self.children = children
        self.files = files
        self.prefix = prefix
        self.is_top_level = is_top_level

        for child in children:
            child.parent = self
        for file in files:
            file.parent = self

    @property
    def path(self):
        return self._path

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
                    File(path)
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
