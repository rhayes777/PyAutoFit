from autofit.mapper.model import ModelInstance, assert_not_frozen
from autofit.mapper.prior.abstract import Prior
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

    def __contains__(self, item):
        return item in self._dict or item in self._dict.values()

    def __getitem__(self, item):
        if item in self._dict:
            return self._dict[item]
        return self.values[item]

    def __len__(self):
        return len(self.values)

    def __str__(self):
        return "\n".join(
            f"{key} = {value}"
            for key, value
            in self.items()
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    @property
    def values(self):
        return list(self._dict.values())

    def items(self):
        return self._dict.items()

    def with_prefix(
            self,
            prefix: str
    ):
        """
        Filter members of the collection, only returning those that start
        with a given prefix as a new collection.
        """
        return CollectionPriorModel({
            key: value
            for key, value
            in self.items()
            if key.startswith(
                prefix
            )
        })

    def as_model(self):
        return CollectionPriorModel({
            key: value.as_model()
            if isinstance(value, AbstractPriorModel)
            else value
            for key, value in self.dict().items()
        })

    def __init__(self, *arguments, **kwargs):
        """
        A prior model used to represent a list of prior models for convenience.

        Arguments are flexibly converted into a collection.

        Parameters
        ----------
        arguments
            Classes, prior models, instances or priors
        """
        super().__init__()
        self.item_number = 0
        arguments = list(arguments)
        if len(arguments) == 0:
            self.add_dict_items(kwargs)
        elif len(arguments) == 1:
            arguments = arguments[0]

            if isinstance(arguments, list):
                for argument in arguments:
                    self.append(argument)
            elif isinstance(arguments, dict):
                self.add_dict_items(arguments)
            else:
                self.append(arguments)
        else:
            self.__init__(arguments)

    @assert_not_frozen
    def add_dict_items(self, item_dict):
        for key, value in item_dict.items():
            if isinstance(key, tuple):
                key = ".".join(key)
            setattr(self, key, AbstractPriorModel.from_object(value))

    def __eq__(self, other):
        if other is None:
            return False
        if len(self) != len(other):
            return False
        for i, item in enumerate(self):
            if item != other[i]:
                return False
        return True

    @assert_not_frozen
    def append(self, item):
        setattr(self, str(self.item_number), AbstractPriorModel.from_object(item))
        self.item_number += 1

    @assert_not_frozen
    def __setitem__(self, key, value):
        obj = AbstractPriorModel.from_object(value)
        try:
            obj.id = getattr(self, str(key)).id
        except AttributeError:
            pass
        setattr(self, str(key), obj)

    @assert_not_frozen
    def __setattr__(self, key, value):
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            try:
                super().__setattr__(key, AbstractPriorModel.from_object(value))
            except AttributeError:
                pass

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
        result = ModelInstance()
        for key, value in self.__dict__.items():
            if isinstance(value, AbstractPriorModel):
                value = value.instance_for_arguments(arguments)
            elif isinstance(value, Prior):
                value = arguments[value]
            setattr(result, key, value)
        return result

    def gaussian_prior_model_for_arguments(self, arguments):
        """
        Create a new collection, updating its priors according to the argument
        dictionary.

        Parameters
        ----------
        arguments
            A dictionary of arguments

        Returns
        -------
        A new collection
        """
        collection = CollectionPriorModel()

        for key, value in self.items():
            if key in (
                    "component_number",
                    "item_number",
                    "id"
            ) or key.startswith(
                "_"
            ):
                continue

            if isinstance(value, AbstractPriorModel):
                collection[key] = value.gaussian_prior_model_for_arguments(
                    arguments
                )
            if isinstance(value, Prior):
                collection[key] = arguments[value]

        return collection

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
