from autofit.mapper.model import ModelInstance
from autofit.mapper.prior import cast_collection, PriorNameValue, ConstantNameValue
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.util import PriorModelNameValue


class CollectionPriorModel(AbstractPriorModel):
    def name_for_prior(self, prior):
        for name, prior_model in self.prior_model_tuples:
            prior_name = prior_model.name_for_prior(prior)
            if prior_name is not None:
                return "{}_{}".format(name, prior_name)

    def __getitem__(self, item):
        return self.items[item]

    def __len__(self):
        return len(self.items)

    @property
    def items(self):
        return [value for key, value in self.__dict__.items() if
                key not in ('component_number', 'item_number', 'id')]

    @property
    def flat_prior_model_tuples(self):
        return [flat_prior_model for prior_model in self.prior_models for
                flat_prior_model in
                prior_model.flat_prior_model_tuples]

    def __init__(self, *arguments, **kwargs):
        """
        A prior model used to represent a list of prior models for convenience.

        Parameters
        ----------
        arguments: list
            A list classes, prior_models or instances
        """
        super().__init__()
        arguments = list(arguments)
        if len(arguments) == 0:
            self.add_dict_items(kwargs)
        elif len(arguments) == 1:
            arguments = arguments[0]

            self.component_number = next(self._ids)
            self.item_number = 0

            if isinstance(arguments, list):
                for argument in arguments:
                    self.append(argument)
            if isinstance(arguments, dict):
                self.add_dict_items(arguments)
        else:
            raise AssertionError("TODO")

    def add_dict_items(self, item_dict):
        for key, value in item_dict.items():
            setattr(self, key, AbstractPriorModel.from_object(value))

    def __add__(self, other):
        new = CollectionPriorModel([])
        for item in self:
            new.append(item)
        for item in other:
            new.append(item)
        return new

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for i, item in enumerate(self):
            if item != other[i]:
                return False
        return True

    def append(self, item):
        setattr(self, str(self.item_number), AbstractPriorModel.from_object(item))
        self.item_number += 1

    def __setitem__(self, key, value):
        obj = AbstractPriorModel.from_object(value)
        try:
            obj.id = getattr(self, str(key)).id
        except AttributeError:
            pass
        setattr(self, str(key), obj)

    def remove(self, item):
        for key, value in self.__dict__.copy().items():
            if value == item:
                del self.__dict__[key]

    @property
    @cast_collection(PriorModelNameValue)
    def label_prior_model_tuples(self):
        return [(prior_model.mapping_name if hasattr(prior_model,
                                                     "mapping_name") else str(i),
                 prior_model) for
                i, prior_model in enumerate(self)]

    @property
    def prior_models(self):
        return [obj for obj in self if isinstance(obj, AbstractPriorModel)]

    def instance_for_arguments(self, arguments):
        """
        Parameters
        ----------
        arguments: {Prior: float}
            A dictionary of arguments

        Returns
        -------
        model_instances: [object]
            A list of instances constructed from the list of prior models.
        """
        result = ModelInstance()
        for key, value in self.__dict__.items():
            if isinstance(value, AbstractPriorModel):
                value = value.instance_for_arguments(arguments)
            setattr(result, key, value)
        return result

    def gaussian_prior_model_for_arguments(self, arguments):
        """
        Parameters
        ----------
        arguments: {Prior: float}
            A dictionary of arguments

        Returns
        -------
        prior_models: [PriorModel]
            A new list of prior models with gaussian priors
        """
        return CollectionPriorModel(
            {
                key: value.gaussian_prior_model_for_arguments(arguments)
                if isinstance(value, AbstractPriorModel)
                else value
                for key, value in self.__dict__.items() if
                key not in ('component_number', 'item_number', 'id')
            }
        )

    @property
    @cast_collection(PriorNameValue)
    def prior_tuples(self):
        """
        Returns
        -------
        priors: [(String, Union(Prior, TuplePrior))]
        """
        return set([prior for prior_model in self.prior_models for prior in
                    prior_model.prior_tuples])

    @property
    @cast_collection(ConstantNameValue)
    def constant_tuples(self):
        """
        Returns
        -------
        priors: [(String, Union(Prior, TuplePrior))]
        """
        return set([constant for prior_model in self.prior_models for constant in
                    prior_model.constant_tuples])

    @property
    def prior_class_dict(self):
        return {prior: cls for prior_model in self.prior_models for prior, cls in
                prior_model.prior_class_dict.items()}
