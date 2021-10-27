import ast
import re
from copy import copy
from pathlib import Path
from typing import Optional

from autofit.tools.edenise.structure.item import Item


class LineItem(Item):
    def __init__(
            self,
            ast_item: ast.stmt,
            parent: Item
    ):
        """
        A line in a file.

        Special consideration is given to lines containing imports.

        When parentheses are not balanced a line can span multiple lines in a file.

        Parameters
        ----------
        ast_item
            The string on the line
        parent
            The file containing the line
        """
        self.ast_item = ast_item
        super().__init__(
            parent=parent
        )

    def __new__(cls, ast_item, parent):
        if isinstance(
                ast_item,
                ast.Import
        ):
            return object.__new__(Import)
        if isinstance(
                ast_item,
                ast.ImportFrom
        ):
            return object.__new__(ImportFrom)
        return object.__new__(LineItem)

    @property
    def children(self):
        """
        Imports don't have any children
        """
        return []

    @property
    def path(self) -> Path:
        return self.parent.path / self.string

    @property
    def name(self) -> str:
        return self.string

    @property
    def is_function(self):
        return re.match(r"def +\w+", self.string) is not None

    @property
    def target_string(self) -> str:
        if self.is_function and self.should_remove_type_annotations:
            result = list()
            open_square_count = 0
            open_bracket_count = 0

            should_add = True
            for i, character in enumerate(self.string):
                def is_window(string):
                    return self.string[i: i + len(string)] == string

                if is_window(" -> ") or is_window(" ->") or is_window("-> ") or is_window("->"):
                    should_add = False
                if character == ":":
                    should_add = open_bracket_count == 0
                if character == "=":
                    should_add = True
                if character == "(":
                    open_bracket_count += 1
                if character == ")":
                    open_bracket_count -= 1
                    if open_bracket_count == 0:
                        should_add = True
                if character == "[":
                    open_square_count += 1
                if character == "]":
                    open_square_count -= 1
                if character in (",", "\n") and open_square_count == 0:
                    should_add = True

                if should_add:
                    result.append(character)

            return "".join(result)
        return self.string


class Import(LineItem):
    @property
    def is_in_project(self) -> bool:
        """
        Is this object within the top level object?
        """
        return any(
            self.ast_item.names[0].name.startswith(
                dependency
            )
            for dependency in self.eden_dependencies
        )

    def converted(self):
        converted = copy(
            self.ast_item
        )
        for name in converted.names:
            if not (self.is_module(name.name) or self.is_member(name.name)):
                name.name = self._edenise_string(
                    name.name
                )
        return converted

    def full_path(self, name):
        return name.split(".")

    def is_module(self, name):
        return self.top_level.is_module(
            self.full_path(
                name
            )
        )

    def is_member(self, name):
        return self.top_level.is_member(
            self.full_path(
                name
            )
        )


class As:
    def __init__(self, item, alias):
        self.item = item
        self.alias = alias

    @property
    def target_import_string(self):
        return f"{self.item.target_import_string} as {self.alias}"


class ImportFrom(Import):
    @property
    def is_in_project(self) -> bool:
        """
        Is this object within the top level object?
        """
        return any(
            self.ast_item.module.startswith(
                dependency
            )
            for dependency in self.eden_dependencies
        )

    def converted(self):
        converted = super().converted()
        converted.level = 0
        converted.module = ".".join(
            map(
                self._edenise_string,
                self.module_path
            )
        )
        return converted

    @property
    def module_path(self):
        path = []
        level = self.ast_item.level

        if level > 0:
            item = self
            for _ in range(level + 1):
                item = item.parent
            items = []
            while item is not None:
                items.append(item)
                item = item.parent

            path += [
                item.name
                for item
                in reversed(items)
            ]

        module = self.ast_item.module
        if module is not None:
            path += self.ast_item.module.split(
                "."
            )
        return path

    def full_path(self, name):
        return self.module_path + super().full_path(name)
