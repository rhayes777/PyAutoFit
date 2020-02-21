from autofit import exc


class Assertion:
    def __init__(
            self,
            lower,
            greater
    ):
        self.lower = lower
        self.greater = greater

    def __call__(self, arg_dict):
        if arg_dict[
            self.lower
        ] > arg_dict[
            self.greater
        ]:
            raise exc.FitException(
                "Assertion failed"
            )
