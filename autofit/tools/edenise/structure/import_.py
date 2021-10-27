import ast
from copy import copy
from typing import Optional

from astunparse import unparse

from autofit.tools.edenise.structure.item import Item


class LineItem(Item):
    def __init__(
            self,
            ast_item: ast.stmt,
            parent: Optional[Item] = None
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

    def converted(self):
        return copy(
            self.ast_item
        )

    @staticmethod
    def parse_fragment(
            string,
            parent=None
    ):
        return LineItem(
            ast.parse(
                string
            ).body[0],
            parent=parent
        )

    @property
    def target_string(self):
        return unparse(
            self.converted()
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
        if isinstance(
                ast_item,
                ast.FunctionDef
        ):
            return object.__new__(Function)
        return object.__new__(LineItem)


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
        converted = super().converted()
        for name in converted.names:
            parts = name.name.split(".")
            converted_path = []
            unconverted_path = []
            for i in range(len(parts)):
                path_prefix = parts[:i + 1]
                total_path = self.module_path + path_prefix
                if not (self.is_module(total_path) or self.is_member(total_path)):
                    converted_path = list(map(
                        self._edenise_string,
                        path_prefix
                    ))
                else:
                    unconverted_path = parts[i:]

            name.name = ".".join(
                converted_path + unconverted_path
            )

        return converted

    @property
    def module_path(self):
        return []

    def is_module(self, path):
        return self.top_level.is_module(
            path
        )

    def is_member(self, path):
        return self.top_level.is_member(
            path
        )


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


class Function(LineItem):
    def converted(self):
        converted = super().converted()
        for arg in converted.args.args:
            arg.annotation = None
        converted.returns = None
        return converted
