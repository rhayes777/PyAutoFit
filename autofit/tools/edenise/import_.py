import ast
from copy import deepcopy, copy
from typing import Optional, Set, List

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
    ) -> "Import":
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
        if len(attribute_names) == 0:
            ast_item = deepcopy(self.ast_item)
            ast_item.names[0].asname = None
            return Import(
                ast_item,
                parent=self.parent
            )

        module = self.ast_item.names[0].name
        level = 0

        if isinstance(
                self,
                ImportFrom
        ):
            if self.ast_item.module is not None:
                module = f"{self.ast_item.module}.{module}"
            level = self.ast_item.level

        # noinspection PyTypeChecker
        return ImportFrom(
            ast.ImportFrom(
                col_offset=self.ast_item.col_offset,
                lineno=self.ast_item.lineno,
                module=module,
                names=[
                    ast.alias(
                        name=attribute_name,
                        asname=None
                    )
                    for attribute_name
                    in sorted(attribute_names)
                ],
                level=level
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

    def converted(self) -> ast.stmt:
        """
        An ast item representing an import converted to comply with Eden
        """
        converted = copy(self.ast_item)
        if self.is_in_project:
            for name in converted.names:
                name.name = ".".join(
                    self.edenise_path(
                        name.name.split("."),
                        prefix=self.module_path
                    )
                )

        return converted

    def edenise_path(
            self,
            path: List[str],
            prefix=None
    ) -> List[str]:
        """
        Convert an import path to eden style. Packages
        which are in the project or a dependency project
        are prefixed and capitalised.

        Parameters
        ----------
        path
            A list of strings defining the path to an import
        prefix
            An optional list of strings defining the first
            part of the path

        Returns
        -------
        A path with silly eden naming conventions applied
        """
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

    def is_module(self, path) -> bool:
        """
        Does the path point to a module?
        """
        return self.top_level.is_module(
            path
        )

    def is_member(self, path):
        """
        Does the path point to an object in a file?
        """
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

    def converted(self) -> ast.stmt:
        """
        An ast item representing an import from converted to comply with eden
        """
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
    def module_path(self) -> List[str]:
        """
        A list of strings describing the import path in the from part of this
        import statement.

        This also accounts for level.

        Examples
        --------
        from autofit.tools import ...
        -> ["autofit", "tools"]

        from .. import ...
        -> ["autofit", "tools"]
        """
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
