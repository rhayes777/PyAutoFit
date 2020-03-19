from autofit.mapper.prior_model.abstract import AbstractPriorModel


class CompoundPrior(AbstractPriorModel):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def instance_for_arguments(self, arguments):
        return self.left.instance_for_arguments(
            arguments
        ) + self.right.instance_for_arguments(
            arguments
        )


class NegativePrior(AbstractPriorModel):
    def __init__(self, prior):
        super().__init__()
        self.prior = prior

    def instance_for_arguments(self, arguments):
        return -self.prior.instance_for_arguments(
            arguments
        )
