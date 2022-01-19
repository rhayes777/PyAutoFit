from .line import LineItem


class Function(LineItem):
    def converted(self):
        converted = super().converted()
        for arg in converted.args.args:
            arg.annotation = None
        converted.returns = None
        return converted
