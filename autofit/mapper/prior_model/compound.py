from autofit.mapper.prior_model import abstract


class CompoundPrior(abstract.AbstractPriorModel):
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
