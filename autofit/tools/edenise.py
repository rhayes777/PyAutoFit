import os
import re
import shutil
from abc import abstractmethod, ABC
from configparser import ConfigParser
from os import walk
from pathlib import Path
from typing import List
from uuid import uuid1


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


class Line:
    def __init__(self, string):
        if "*" in string:
            print("Please ensure no imports in the __init__ contain a *")
            exit(1)
        if "," in string:
            print("Comma separated imports not allowed")
            exit(1)
        self.string = string.replace("\n", "")
        self.id = str(uuid1())

    @property
    def sources(self):
        return (
            self.source,
            self.joint_source
        )

    @property
    def joint_source(self):
        return f"{self.import_target[1:]}.{self.source}"

    def __str__(self):
        return f"{self.source} -> {self.import_target}"

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    def __len__(self):
        return len(self.source)

    def __lt__(self, other):
        return len(self) < len(other)

    def __gt__(self, other):
        return len(self) > len(other)

    def __eq__(self, other):
        return self._target == other._target

    @property
    def is_import(self):
        return self.string.startswith("from")

    @property
    def source(self):
        match = re.match("from .* as (.+)", self.string)
        if match is not None:
            return match.group(1)
        return re.match("from .* import (.+)", self.string).group(1)

    @property
    def target(self):
        return self._target.split(".")[-1]

    @property
    def import_target(self):
        target = ".".join(
            self._target.split(".")[:-1]
        )
        if len(target) > 0:
            return f".{target}"
        return target

    def __hash__(self):
        return hash(self._target)

    @property
    def _target(self):
        return self.string.replace(
            f" as {self.source}",
            ""
        ).replace(
            "from ",
            ""
        ).replace(
            " import ",
            "."
        ).lstrip(
            "."
        )


class Converter:
    def __init__(self, name, prefix, lines):
        self.name = name
        self.prefix = prefix
        self.lines = sorted(
            filter(
                lambda line: line.is_import,
                lines
            ),
            reverse=True
        )

    @classmethod
    def from_prefix_and_source_directory(
            cls,
            name,
            prefix,
            source_directory
    ):
        source_directory = source_directory
        with open(
                f"{source_directory}/__init__.py"
        ) as f:
            lines = map(Line, f.readlines())
        return Converter(name, prefix, lines)

    def convert(self, string):
        matched_lines = set()
        string = string.replace(f"import {self.name} as {self.prefix}", "")
        for line in self.lines:
            for line_source in line.sources:
                source = f"{self.prefix}.{line_source}"
                if source in string:
                    matched_lines.add(
                        line
                    )
                    string = string.replace(
                        source,
                        line.id
                    )
        for line in self.lines:
            string = string.replace(
                line.id,
                line.target
            )

        import_string = "\n".join(
            f"from {self.name}{line.import_target} import {line.target}"
            for line in matched_lines
        )
        return f"{import_string}\n{string}"


def edenise_directory(
        root_directory
):
    try:
        config = ConfigParser()
        config.read(
            f"{root_directory}/eden.ini"
        )

        edenise(
            root_directory,
            config.get("eden", "name"),
            config.get("eden", "prefix")
        )
    except ValueError:
        print("Usage: ./edenise.py root_directory")
        exit(1)


def edenise(
        root_directory,
        name,
        prefix
):
    target_directory = f"{root_directory}/../eden/{name}_eden"

    print(f"Creating {target_directory}...")
    shutil.copytree(
        root_directory,
        target_directory,
        symlinks=True
    )

    converter = Converter.from_prefix_and_source_directory(
        name=name,
        prefix=prefix,
        source_directory=f"{root_directory}/{name}"
    )

    for root, _, files in walk(f"{target_directory}/test_{name}"):
        for file in files:
            if file.endswith(".py"):
                with open(f"{root}/{file}", "r+") as f:
                    string = f.read()
                    f.seek(0)
                    f.write(
                        converter.convert(
                            string
                        )
                    )
                    f.truncate()

    for root, _, files in walk(f"{target_directory}/{name}"):
        try:
            for file in ["mock.py", "mock_real.py"]:
                with open(f"{root}/{file}", "r+") as f:
                    string = f.read()
                    f.seek(0)
                    f.write(
                        converter.convert(
                            string
                        )
                    )
                    f.truncate()
        except FileNotFoundError:
            continue

    open(f"{target_directory}/{name}/__init__.py", "w+").close()
