from autofit.mapper.prior import Promise
from autofit.tools.phase import break_promises


class SomeClass:
    def __init__(self, a, b):
        self.a = a
        self.b = b


def test_break_promises():
    cls = SomeClass(Promise(None, result_path=(), assert_exists=False), "b")
    cls = break_promises(cls)
    assert cls.a is None
    assert cls.b == "b"
