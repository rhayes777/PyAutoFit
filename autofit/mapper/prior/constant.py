from autofit.mapper.variable import Variable


class Constant(Variable):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.value == other
        return super().__eq__(other)

    def dict(self):
        return {"type": "Constant", "value": self.value}
