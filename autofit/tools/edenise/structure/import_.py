import ast
from typing import Optional, Set

from .item import Item
from .line import LineItem


class Import(LineItem):
    def __init__(
            self,
            ast_item: ast.stmt,
            parent: Optional[Item] = None
    ):
        super().__init__(ast_item, parent)

    @property
    def is_aliased(self) -> bool:
        """
        True if the import is of the form

        import something as alias
        """
        return self.alias is not None

    @property
    def alias(self):
        for name in self.ast_item.names:
            return name.asname

    def as_from_import(
            self,
            attribute_names: Set[str]
    ) -> "ImportFrom":
        """
        Convert an import as import to an explicit
        import of each attribute.

        Parameters
        ----------
        attribute_names
            A list of names of attributes accessed on
            the alias

        Returns
        -------
        An explicit import of each attribute

        Examples
        --------
        import autofit as af

        af.Model(af.Gaussian)

        becomes

        from autofit import Model, Gaussian

        Model(Gaussian)
        """
        # noinspection PyTypeChecker
        return ImportFrom(
            ast.ImportFrom(
                col_offset=self.ast_item.col_offset,
                lineno=self.ast_item.lineno,
                module=self.ast_item.names[0].name,
                names=[
                    ast.alias(
                        name=attribute_name,
                        asname=None
                    )
                    for attribute_name
                    in sorted(attribute_names)
                ],
                level=0
            ),
            parent=self.parent
        )

    @property
    def is_in_project(self) -> bool:
        """
        Is this object within the top level object?
        """
        return self.top_level.is_in_project(
            self.ast_item.names[0].name.split(".")
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
        return self.top_level.is_in_project(
            self.ast_item.module.split(".")
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
