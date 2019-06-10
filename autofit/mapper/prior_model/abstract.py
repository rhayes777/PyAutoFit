import inspect

from autofit.mapper.prior import Prior, Constant
from autofit.mapper.model_object import ModelObject
from autofit.mapper.prior import cast_collection, PriorNameValue, ConstantNameValue
from autofit.mapper.prior_model.util import PriorModelNameValue


class AbstractPriorModel(ModelObject):
    """
    Abstract model that maps a set of priors to a particular class. Must be
    overridden by any prior model so that the model mapper recognises its prior \
    model attributes.

    @DynamicAttrs
    """

    @property
    def name(self):
        return self.__class__.__name__

    # noinspection PyUnusedLocal
    @staticmethod
    def from_object(t, *args, **kwargs):
        if inspect.isclass(t):
            from .prior_model import PriorModel
            obj = object.__new__(PriorModel)
            obj.__init__(t, **kwargs)
        elif isinstance(t, list) or isinstance(t, dict):
            from autofit.mapper import CollectionPriorModel
            obj = object.__new__(CollectionPriorModel)
            obj.__init__(t)
        else:
            obj = t
        return obj

    @property
    def info(self):
        info = []

        prior_model_iterator = self.direct_prior_tuples + self.direct_constant_tuples

        for attribute_tuple in prior_model_iterator:
            attribute = attribute_tuple[1]

            # noinspection PyUnresolvedReferences
            line = attribute_tuple.name
            # noinspection PyUnresolvedReferences
            info.append(line + ' ' * (60 - len(line)) + attribute.info)

        for prior_model_name, prior_model in self.prior_model_tuples:
            info.append(prior_model.name + '\n')
            info.extend([f"{prior_model_name}_{item}" for item in prior_model.info])

        return info

    @property
    @cast_collection(PriorNameValue)
    def direct_prior_tuples(self):
        return self.tuples_with_type(Prior)

    @property
    @cast_collection(ConstantNameValue)
    def direct_constant_tuples(self):
        return self.tuples_with_type(Constant)

    @property
    def flat_prior_model_tuples(self):
        """
        Returns
        -------
        prior_models: [(str, AbstractPriorModel)]
            A list of prior models associated with this instance
        """
        raise NotImplementedError(
            "PriorModels must implement the flat_prior_models property")

    @property
    @cast_collection(PriorModelNameValue)
    def prior_model_tuples(self):
        return self.tuples_with_type(AbstractPriorModel)

    @property
    def prior_tuples(self):
        raise NotImplementedError()

    @property
    @cast_collection(PriorModelNameValue)
    def direct_prior_model_tuples(self):
        return [(name, value) for name, value in self.__dict__.items() if
                isinstance(value, AbstractPriorModel)]

    def __eq__(self, other):
        return isinstance(other, AbstractPriorModel) \
               and self.direct_prior_model_tuples == other.direct_prior_model_tuples

    @property
    def constant_tuples(self):
        raise NotImplementedError()

    @property
    def prior_class_dict(self):
        raise NotImplementedError()

    def instance_for_arguments(self, arguments):
        raise NotImplementedError()

    @property
    def prior_count(self):
        return len(self.prior_tuples)

    @property
    def priors(self):
        return [prior_tuple.prior for prior_tuple in self.prior_tuples]

    def name_for_prior(self, prior):
        for prior_model_name, prior_model in self.direct_prior_model_tuples:
            prior_name = prior_model.name_for_prior(prior)
            if prior_name is not None:
                return "{}_{}".format(prior_model_name, prior_name)
        prior_tuples = self.prior_tuples
        for name, p in prior_tuples:
            if p == prior:
                return name

    def __hash__(self):
        return self.id

    def tuples_with_type(self, class_type):
        return list(filter(lambda t: t[0] != "id" and isinstance(t[1], class_type),
                           self.__dict__.items()))
