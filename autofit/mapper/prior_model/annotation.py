from autofit.mapper.prior_model.prior_model import PriorModel, Prior


class AnnotationPriorModel(PriorModel):
    def __init__(self, cls, parent_class, true_argument_name, **kwargs):
        self.parent_class = parent_class
        self.true_argument_name = true_argument_name
        super().__init__(cls, **kwargs)

    def make_prior(self, attribute_name):
        return Prior.for_class_and_attribute_name(
            self.parent_class, self.true_argument_name
        )
