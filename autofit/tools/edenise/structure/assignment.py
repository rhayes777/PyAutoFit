import ast

from .line import LineItem


class Assignment(LineItem):
    def converted(self):
        converted = super().converted()
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
        return converted
