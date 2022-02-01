import ast
from copy import copy
from typing import Optional, List

from astunparse import unparse

from .item import Item


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

    @property
    def children(self) -> List["LineItem"]:
        if hasattr(
                self.ast_item,
                "body"
        ):
            return [
                LineItem(
                    item,
                    parent=self.parent
                )
                for item
                in self.ast_item.body
            ]
        return []

    def converted(self) -> ast.stmt:
        """
        Convert this line, producing a new ast statement that conforms
        to the arbitrary and sadistic Eden requirements.
        """
        converted = copy(self.ast_item)
        aliases = self.parent.aliases

        def strip_ids(
                obj
        ):
            if isinstance(
                    obj, ast.Attribute
            ) and hasattr(
                obj.value, "id"
            ) and obj.value.id in aliases:
                return ast.Name(
                    id=obj.attr,
                    col_offset=obj.col_offset,
                    ctx=obj.ctx,
                    lineno=obj.lineno
                )
            if hasattr(obj, "__dict__"):
                obj.__dict__ = strip_ids(
                    obj.__dict__
                )
            if isinstance(obj, dict):
                return {
                    key: strip_ids(
                        value
                    )
                    for key, value
                    in obj.items()
                }
            if isinstance(obj, list):
                return list(map(
                    strip_ids,
                    obj
                ))
            return obj

        strip_ids(converted)

        if len(self.children) > 0:
            converted.body = [
                child.converted()
                for child
                in self.children
            ]

        return converted

    @staticmethod
    def parse_fragment(
            string: str,
            parent=None
    ) -> "LineItem":
        """
        Parse from a string. Used for testing.
        """
        return LineItem(
            ast.parse(
                string
            ).body[0],
            parent=parent
        )

    @property
    def target_string(self) -> str:
        """
        The string made to please eden.

        Used for testing.
        """
        return unparse(
            self.converted()
        )

    def __new__(cls, ast_item, parent):
        """
        Factory pattern determines which class to instantiate based
        on the type of the ast statement.
        """
        from .import_ import Import, ImportFrom
        from .function import Function

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
