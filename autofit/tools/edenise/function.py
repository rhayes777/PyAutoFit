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

        body = list()
        for line in self.lines():
            if isinstance(
                    line, Import
            ) and line.is_in_project and line.is_aliased:
                line = line.as_from_import(
                    self.parent.attributes_for_alias(
                        line.alias
                    )
                )
            body.append(
                line.converted()
            )
        converted.body = body
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
