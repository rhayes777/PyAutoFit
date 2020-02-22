from autofit import AttributeNameValue


class PriorModelNameValue(AttributeNameValue):
    @property
    def prior_model(self):
        return self.value
