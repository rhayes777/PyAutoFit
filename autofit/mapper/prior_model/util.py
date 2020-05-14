from autofit.mapper.prior_model.attribute_pair import AttributeNameValue


class PriorModelNameValue(AttributeNameValue):
    @property
    def prior_model(self):
        return self.value
