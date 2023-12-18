from collections.abc import Iterable

from autofit.jax_wrapper import register_pytree_node_class

from autofit.mapper.model import ModelInstance, assert_not_frozen
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior_model.abstract import AbstractPriorModel


@register_pytree_node_class
class Collection(AbstractPriorModel):
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

    def tree_flatten(self):
        keys, values = zip(*self.items())
        return values, keys

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        instance = cls()

        for key, value in zip(aux_data, children):
            setattr(instance, key, value)
        return instance

    def __contains__(self, item):
        return item in self._dict or item in self._dict.values()

    def __getitem__(self, item):
        if item in self._dict:
            return self._dict[item]
        return self.values[item]

    def __len__(self):
        return len(self.values)

    def __str__(self):
        return "\n".join(f"{key} = {value}" for key, value in self.items())

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    @property
    def values(self):
        return list(self._dict.values())

    def items(self):
        return self._dict.items()

    def with_prefix(self, prefix: str):
        """
        Filter members of the collection, only returning those that start
        with a given prefix as a new collection.
        """
        return Collection(
            {key: value for key, value in self.items() if key.startswith(prefix)}
        )

    def as_model(self):
        return Collection(
            {
                key: value.as_model()
                if isinstance(value, AbstractPriorModel)
                else value
                for key, value in self.dict().items()
            }
        )

    def __init__(self, *arguments, **kwargs):
        """
        The object multiple Python classes are input into to create model-components, which has free parameters that
        are fitted by a non-linear search.

        Multiple Python classes can be input into a `Collection` in order to compose high dimensional models made of
        multiple model-components.

        The ``Collection`` object is highly flexible, and can create models from many input Python data structures
        (e.g. a list of classes, dictionary of classes, hierarchy of classes).

        For a complete description of the model composition API, see the **PyAutoFit** model API cookbooks:

        https://pyautofit.readthedocs.io/en/latest/cookbooks/cookbook_1_basics.html

        The Python class input into a ``Model`` to create a model component is written using the following format:

        - The name of the class is the name of the model component (e.g. ``Gaussian``).
        - The input arguments of the constructor are the parameters of the mode (e.g. ``centre``, ``normalization`` and ``sigma``).
        - The default values of the input arguments tell PyAutoFit whether a parameter is a single-valued float or a
        multi-valued tuple.

        [Rich document more clearly]

        A prior model used to represent a list of prior models for convenience.

        Arguments are flexibly converted into a collection.

        Parameters
        ----------
        arguments
            Classes, prior models, instances or priors

        Examples
        --------

        class Gaussian:

            def __init__(
                self,
                centre=0.0,        # <- PyAutoFit recognises these
                normalization=0.1, # <- constructor arguments are
                sigma=0.01,        # <- the Gaussian's parameters.
            ):
                self.centre = centre
                self.normalization = normalization
                self.sigma = sigma

        model = af.Collection(gaussian_0=Gaussian, gaussian_1=Gaussian)
        """
        super().__init__()
        self.item_number = 0
        arguments = list(arguments)
        if len(arguments) == 0:
            self.add_dict_items(kwargs)
        elif len(arguments) == 1:
            arguments = arguments[0]

            if isinstance(arguments, dict):
                self.add_dict_items(arguments)
            elif isinstance(arguments, Iterable):
                for argument in arguments:
                    self.append(argument)
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
        collection = Collection()

        for key, value in self.items():
            if key in ("component_number", "item_number", "id") or key.startswith("_"):
                continue

            if isinstance(value, AbstractPriorModel):
                collection[key] = value.gaussian_prior_model_for_arguments(arguments)
            if isinstance(value, Prior):
                collection[key] = arguments[value]

        return collection

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
            if key.startswith("_"):
                continue
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
        collection = Collection()

        for key, value in self.items():
            if key in ("component_number", "item_number", "id") or key.startswith("_"):
                continue

            if isinstance(value, AbstractPriorModel):
                collection[key] = value.gaussian_prior_model_for_arguments(arguments)
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
