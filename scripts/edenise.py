#!/usr/bin/env python

import re
import shutil
from os import path, walk

import pytest


class Line:
    def __init__(self, string):
        if "*" in string:
            print("Please ensure no imports in the __init__ contain a *")
            exit(1)
        self.string = string.replace("\n", "")

    def __str__(self):
        return f"{self.source} -> {self.target}"

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

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
        self.lines = list(filter(
            lambda line: line.is_import,
            lines
        ))

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
            target = f"{self.prefix}.{line.target}"
            string = string.replace(
                source,
                target
            )
        return string


@pytest.fixture(
    name="as_line"
)
def make_as_line():
    return Line(
        "from .mapper.model import ModelInstance as Instance"
    )


@pytest.fixture(
    name="line"
)
def make_line():
    return Line(
        "from .mapper.model import ModelInstance"
    )


# class Test:
#     def test_line_is_import(self, as_line):
#         assert as_line.is_import
#         assert not Line(".mapper.model").is_import
#
#     def test_source(self, as_line, line):
#         assert as_line.source == "Instance"
#         assert line.source == "ModelInstance"
#
#     def test_target(self, as_line, line):
#         assert as_line.target == "mapper.model.ModelInstance"
#         assert line.target == "mapper.model.ModelInstance"
#
#     def test_replace(self, as_line, line):
#         converter = Converter(
#             "af",
#             [as_line, line]
#         )
#         assert converter.convert(
#             "af.ModelInstance\naf.Instance"
#         ) == "af.mapper.model.ModelInstance\naf.mapper.model.ModelInstance"


def main():
    root_directory = f"{path.dirname(path.realpath(__file__))}/.."

    name = "autofit"
    prefix = "af"

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


if __name__ == "__main__":
    main()
