from typing import List, cast

from .import_ import Import
from .line import LineItem


class Function(LineItem):
    def lines(self):
        return [
            LineItem(
                item,
                parent=self.parent
            )
            for item
            in self.ast_item.body
        ]

    def converted(self):
        converted = super().converted()
        for arg in converted.args.args:
            arg.annotation = None
        converted.returns = None
        converted.body = [
            item.converted()
            for item in self.lines()
        ]
        return converted

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
