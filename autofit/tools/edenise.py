import re
import shutil
from os import walk
from uuid import uuid1


class Line:
    def __init__(self, string):
        if "*" in string:
            print("Please ensure no imports in the __init__ contain a *")
            exit(1)
        self.string = string.replace("\n", "")
        self.id = str(uuid1())

    def __str__(self):
        return f"{self.source} -> {self.target}"

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    def __len__(self):
        return len(self.source)

    def __lt__(self, other):
        return len(self) < len(other)

    def __gt__(self, other):
        return len(self) > len(other)

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
    def __init__(self, prefix, lines):
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
            prefix,
            source_directory
    ):
        source_directory = source_directory
        with open(
                f"{source_directory}/__init__.py"
        ) as f:
            lines = map(Line, f.readlines())
        return Converter(prefix, lines)

    def convert(self, string):
        for line in self.lines:
            source = f"{self.prefix}.{line.source}"
            string = string.replace(
                source,
                line.id
            )
        for line in self.lines:
            target = f"{self.prefix}.{line.target}"
            string = string.replace(
                line.id,
                target
            )
        return string


def edenise(
        root_directory,
        name,
        prefix
):
    target_directory = f"{root_directory}/../{name}_eden"

    print(f"Creating {target_directory}...")
    shutil.copytree(
        root_directory,
        target_directory,
        symlinks=True
    )

    converter = Converter.from_prefix_and_source_directory(
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
