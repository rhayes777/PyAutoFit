import ast
from typing import Optional

from .line import LineItem
from .item import Item


class Import(LineItem):
    def __init__(
            self,
            ast_item: ast.stmt,
            parent: Optional[Item] = None
    ):
        super().__init__(ast_item, parent)

    @property
    def is_aliased(self):
        return self.alias is not None

    @property
    def alias(self):
        for name in self.ast_item.names:
            return name.asname

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
        if self.is_in_project:
            for name in converted.names:
                name.name = ".".join(
                    self.edenise_path(
                        name.name.split("."),
                        prefix=self.module_path
                    )
                )

        return converted

    def edenise_path(self, path, prefix=None):
        converted_path = []
        unconverted_path = []
        for i in range(len(path)):
            path_prefix = path[:i + 1]
            total_path = (prefix or []) + path_prefix
            if not self.is_module(total_path) and not self.is_member(total_path):
                converted_path = list(map(
                    self._edenise_string,
                    path_prefix
                ))
            else:
                unconverted_path = path[i:]

        return converted_path + unconverted_path

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
        if self.ast_item.level > 0:
            return True
        return any(
            self.ast_item.module.startswith(
                dependency
            )
            for dependency in self.eden_dependencies
        )

    def converted(self):
        converted = super().converted()
        if self.is_in_project:
            converted.level = 0
            converted.module = ".".join(
                self.edenise_path(
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


