import ast
from copy import copy
from typing import Optional

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
        from .import_ import Import, ImportFrom
        from .function import Function
        from .assignment import Assignment

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
        if isinstance(
                ast_item,
                ast.Assign
        ):
            return object.__new__(Assignment)
        return object.__new__(LineItem)
