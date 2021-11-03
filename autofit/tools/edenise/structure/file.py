import ast
from pathlib import Path
from typing import List, cast

from astunparse import unparse

from .import_ import Import
from .item import Item, DirectoryItem
from .line import LineItem


class File(DirectoryItem):
    def __init__(
            self,
            path: Path,
            prefix: str,
            parent: Item
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
        super().__init__(
            prefix,
            parent=parent
        )
        self._path = path
        self._ast_item = None

    @property
    def aliased_imports(self):
        return [
            import_
            for import_
            in self.imports
            if import_.is_aliased
        ]

    @property
    def aliases(self):
        return [
            import_.alias
            for import_
            in self.aliased_imports
        ]

    def attributes_for_alias(
            self,
            alias: str
    ):
        return {
            attribute.attr
            for attribute
            in self.attributes()
            if attribute.value.id == alias
        }

    def attributes(self):
        def get_attributes(
                obj
        ):
            if isinstance(
                    obj, ast.Attribute
            ):
                yield obj
            items = []
            if hasattr(obj, "__dict__"):
                items = obj.__dict__.values()
            if isinstance(obj, dict):
                items = obj.values()
            if isinstance(obj, list):
                items = obj
            for item in items:
                for attribute in get_attributes(
                        item
                ):
                    yield attribute

        return get_attributes(
            self.ast_item
        )

    @property
    def ast_item(self):
        if self._ast_item is None:
            with open(self.path) as f:
                self._ast_item = ast.parse(
                    f.read()
                )
        return self._ast_item

    def generate_target(self, output_path):
        with open(output_path / self.target_path, "w+") as f:
            f.write(self.target_string)

    @property
    def target_string(self):
        return unparse(self.converted())

    def converted(self):
        module = ast.Module()
        converted_lines = []

        for line in self.lines():
            if isinstance(
                    line, Import
            ) and line.is_aliased:
                line = line.as_from_import(
                    self.attributes_for_alias(
                        line.alias
                    )
                )
            converted_lines.append(
                line.converted()
            )

        module.body = converted_lines
        return module

    @property
    def target_name(self) -> str:
        """
        The name of this file after edenisation
        """
        if not self.should_rename_modules or self.name == "__init__":
            return self.name
        return super().target_name

    @property
    def target_file_name(self) -> str:
        return f"{self.target_name}.py"

    @property
    def target_import_string(self) -> str:
        """
        The string for importing this file after edenisation
        """
        string = self.target_name.replace(".py", "")
        if self.should_rename_modules:
            string = f"{string} as {self.name}"
        return string

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

    def lines(self):
        """
        Yield objects comprising 'lines'

        Open parentheses are used to determine if a 'line' goes across several
        true lines
        """
        with open(self.path) as f:
            return [
                LineItem(item, self)
                for item in ast.parse(
                    f.read()
                ).body
            ]

    @property
    def imports(self) -> List[Import]:
        """
        Imports in the file
        """
        return cast(
            List[Import],
            list(filter(
                lambda item: isinstance(
                    item, Import
                ),
                self.lines()
            ))
        )

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
