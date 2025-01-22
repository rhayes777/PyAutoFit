import autofit as af


class Class:
    def __init__(self, argument: float = 0.0):
        self.argument = argument


def test_placeholder():
    model = af.Model(Class)

    assert (
        model.info
        == """Total Free Parameters = 0



argument                                                                        Prior Missing: Enter Manually or Add to Config"""
    )
