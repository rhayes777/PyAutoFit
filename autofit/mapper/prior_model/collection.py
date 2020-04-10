from autofit import exc
from autofit.mapper.prior import Prior
from autofit.mapper.model import ModelInstance
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.abstract import check_assertions


class CollectionPriorModel(AbstractPriorModel):
    def name_for_prior(self, prior: Prior) -> str:
        """
        Construct a name for the prior. This is the path taken
        to get to the prior.

        Parameters
        ----------
        prior

        Returns
        -------
        A string of object names joined by underscores
        """
        for name, prior_model in self.prior_model_tuples:
            prior_name = prior_model.name_for_prior(prior)
            if prior_name is not None:
                return "{}_{}".format(name, prior_name)
        for name, direct_prior in self.direct_prior_tuples:
            if prior == direct_prior:
                return name

    def __getitem__(self, item):
        return self.values[item]

    def __len__(self):
        return len(self.values)

    @property
    def dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("component_number", "item_number", "id")
               and not key.startswith("_")
        }

    @property
    def values(self):
        return list(self.dict.values())

    def items(self):
        return self.dict.items()

    def as_model(self):
        return CollectionPriorModel(
            {
                key: value.as_model()
                if isinstance(value, AbstractPriorModel)
                else value
                for key, value in self.dict.items()
            }
        )

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

    def __setattr__(self, key, value):
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            super().__setattr__(key, AbstractPriorModel.from_object(value))

    def remove(self, item):
        for key, value in self.__dict__.copy().items():
            if value == item:
                del self.__dict__[key]

    @check_assertions
    def _instance_for_arguments(self, arguments):
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
        if self.promise_count > 0:
            raise exc.PriorException(
                "All promises must be populated prior to instantiation"
            )
        result = ModelInstance()
        for key, value in self.__dict__.items():
            if isinstance(value, AbstractPriorModel):
                value = value.instance_for_arguments(arguments)
            if isinstance(value, Prior):
                value = value.value_for(arguments[value])
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
        return CollectionPriorModel({
            key: value.gaussian_prior_model_for_arguments(arguments)
            if isinstance(value, AbstractPriorModel)
            else value
            for key, value in self.__dict__.items()
            if key not in ("component_number", "item_number", "id") and not key.startswith(
                "_"
            )
        })

    @property
    def prior_class_dict(self):
        return {
            **{
                prior: cls
                for prior_model in self.direct_prior_model_tuples
                for prior, cls in prior_model[1].prior_class_dict.items()
            },
            **{prior: ModelInstance for _, prior in self.direct_prior_tuples},
        }
