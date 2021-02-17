import autofit as af
from autofit.mock.mock import Gaussian


class Sampling(af.AbstractPriorModel):
    def __init__(
            self,
            model,
            method,
            **kwargs
    ):
        super().__init__()
        self.model = model
        self.method = method
        self.kwargs = kwargs

    def _instance_for_arguments(self, arguments):
        # noinspection PyProtectedMember
        instance = self.model._instance_for_arguments(
            arguments
        )
        return self.method(
            instance,
            **{
                key: value.random()
                if isinstance(
                    value,
                    af.Prior
                )
                else value
                for key, value
                in self.kwargs.items()
            }
        )


def test():
    samples = Sampling(
        af.PriorModel(
            Gaussian
        ),
        Gaussian.inverse,
        y=af.UniformPrior(
            lower_limit=0.0,
            upper_limit=1.0
        )
    )

    assert isinstance(
        samples.instance_from_prior_medians(),
        float
    )
