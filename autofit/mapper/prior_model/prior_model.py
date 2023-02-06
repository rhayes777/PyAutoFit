import builtins
import copy
import inspect
import logging

from jax._src.tree_util import register_pytree_node_class, register_pytree_node

from autoconf.class_path import get_class_path
from autoconf.exc import ConfigException
from autofit.mapper.model import assert_not_frozen
from autofit.mapper.model_object import ModelObject
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior.deferred import DeferredInstance
from autofit.mapper.prior.tuple_prior import TuplePrior
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.abstract import check_assertions
from autofit.tools.namer import namer

logger = logging.getLogger(__name__)

class_args_dict = dict()


@register_pytree_node_class
class Model(AbstractPriorModel):
    """
    @DynamicAttrs
    """

    @property
    def name(self):
        return self.cls.__name__

    def __str__(self):
        prior_string = ", ".join(map(str, self.prior_tuples))
        return f"{self.name} {prior_string}"

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    def as_model(self):
        return Model(self.cls)

    def __hash__(self):
        return self.id

    def __add__(self, other):
        if self.cls != other.cls:
            raise TypeError(
                f"Cannot add PriorModels with different classes "
                f"({self.cls.__name__} and {other.cls.__name__})"
            )
        return super().__add__(other)

    def __init__(self, cls, **kwargs):
        """
        The object a Python class is input into to create a model-component, which has free parameters that are fitted
        by a non-linear search.

        The ``Model`` object is flexible, and can create models from many input Python data structures
        (e.g. a list of classes, dictionary of classes, hierarchy of classes).

        For a complete description of the model composition API, see the **PyAutoFit** model API cookbooks:

        https://pyautofit.readthedocs.io/en/latest/cookbooks/cookbook_1_basics.html

        The Python class input into a ``Model`` to create a model component is written using the following format:

        - The name of the class is the name of the model component (e.g. ``Gaussian``).
        - The input arguments of the constructor are the parameters of the mode (e.g. ``centre``, ``normalization`` and ``sigma``).
        - The default values of the input arguments tell PyAutoFit whether a parameter is a single-valued float or a
        multi-valued tuple.

        [Rich explain everything else]

        Parameters
        ----------
        cls
            The class associated with this instance

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

        model = af.Model(Gaussian)
        """
        super().__init__(label=namer(cls.__name__) if inspect.isclass(cls) else None)
        if cls is self:
            return

        if not (inspect.isclass(cls) or inspect.isfunction(cls)):
            raise AssertionError(f"{cls} is not a class or function")

        self.cls = cls

        try:
            annotations = inspect.getfullargspec(cls).annotations
            for key, value in annotations.items():
                if isinstance(value, str):
                    annotations[key] = getattr(builtins, value)
        except TypeError:
            annotations = dict()

        try:
            arg_spec = inspect.getfullargspec(cls)
            defaults = dict(
                zip(arg_spec.args[-len(arg_spec.defaults) :], arg_spec.defaults)
            )
        except TypeError:
            defaults = {}

        args = self.constructor_argument_names

        if "settings" in defaults:
            del defaults["settings"]
        if "settings" in args:
            args.remove("settings")

        for arg in args:
            if isinstance(defaults.get(arg), str):
                continue

            if arg in kwargs:
                keyword_arg = kwargs[arg]
                if isinstance(keyword_arg, (list, dict)):
                    from autofit.mapper.prior_model.collection import Collection

                    ls = Collection(keyword_arg)

                    setattr(self, arg, ls)
                else:
                    if inspect.isclass(keyword_arg):
                        keyword_arg = Model(keyword_arg)
                    setattr(self, arg, keyword_arg)
            elif arg in defaults and isinstance(defaults[arg], tuple):
                tuple_prior = TuplePrior()
                for i in range(len(defaults[arg])):
                    attribute_name = "{}_{}".format(arg, i)
                    setattr(
                        tuple_prior, attribute_name, self.make_prior(attribute_name)
                    )
                setattr(self, arg, tuple_prior)
            elif arg in annotations and annotations[arg] is not float:
                spec = annotations[arg]

                # noinspection PyUnresolvedReferences
                if inspect.isclass(spec) and issubclass(spec, float):
                    from autofit.mapper.prior_model.annotation import (
                        AnnotationPriorModel,
                    )

                    setattr(self, arg, AnnotationPriorModel(spec, cls, arg))
                elif hasattr(spec, "__args__") and type(None) in spec.__args__:
                    setattr(self, arg, None)
                else:
                    setattr(self, arg, Model(annotations[arg]))
            else:
                prior = self.make_prior(arg)
                if (
                    isinstance(prior, ConfigException)
                    and hasattr(cls, "__default_fields__")
                    and arg in cls.__default_fields__
                ):
                    prior = defaults[arg]
                setattr(self, arg, prior)
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, Model(value) if inspect.isclass(value) else value)

        try:
            # noinspection PyTypeChecker
            register_pytree_node(
                self.cls, self.instance_flatten, self.instance_unflatten,
            )
        except ValueError:
            pass

    def instance_flatten(self, instance):
        return (
            [getattr(instance, name) for name in self.constructor_argument_names],
            None,
        )

    def instance_unflatten(self, aux_data, children):
        return self.cls(**dict(zip(self.constructor_argument_names, children)))

    def tree_flatten(self):
        names, priors = zip(*self.direct_prior_tuples)
        return priors, (names, self.cls)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        names, cls_ = aux_data
        arguments = {name: child for name, child in zip(names, children)}
        return cls(cls_, **arguments)

    def dict(self):
        return {"class_path": get_class_path(self.cls), **super().dict()}

    # noinspection PyAttributeOutsideInit
    @property
    def constructor_argument_names(self):
        if self.cls not in class_args_dict:
            try:
                class_args_dict[self.cls] = inspect.getfullargspec(self.cls).args[1:]
            except TypeError:
                class_args_dict[self.cls] = []
        return class_args_dict[self.cls]

    def __eq__(self, other):
        return (
            isinstance(other, Model)
            and self.cls == other.cls
            and self.prior_tuples == other.prior_tuples
        )

    def make_prior(self, attribute_name):
        """
        Returns a prior for an attribute of a class with a given name. The prior is
        created by searching the default prior config for the attribute.

        Entries in configuration with a u become uniform priors; with a g become
        gaussian priors; with a c become instances.

        If prior configuration for a given attribute is not specified in the
        configuration for a class then the configuration corresponding to the parents
        of that class is searched. If no configuration can be found then a prior
        exception is raised.

        Parameters
        ----------
        attribute_name: str
            The name of the attribute for which a prior is created

        Returns
        -------
        prior: p.Prior
            A prior

        Raises
        ------
        exc.PriorException
            If no configuration can be found
        """
        cls = self.cls
        if not inspect.isclass(cls):
            # noinspection PyProtectedMember
            cls = inspect._findclass(cls)
        try:
            return Prior.for_class_and_attribute_name(cls, attribute_name)
        except ConfigException as e:
            return e

    @assert_not_frozen
    def __setattr__(self, key, value):
        try:
            value.label = namer(key)
        except (AttributeError, TypeError):
            pass

        if key not in (
            "component_number",
            "phase_property_position",
            "mapping_name",
            "id",
            "_is_frozen",
            "_frozen_cache",
        ):
            try:
                if "_" in key:
                    name = key.split("_")[0]
                    tuple_prior = [v for k, v in self.tuple_prior_tuples if name == k][
                        0
                    ]
                    setattr(tuple_prior, key, value)
                    return
            except IndexError:
                pass
        try:
            super().__setattr__(key, value)
        except AttributeError as e:
            logger.exception(e)
            logger.exception(key)

    def __getattr__(self, item):
        try:
            if (
                "_" in item
                and item not in ("_is_frozen", "tuple_prior_tuples")
                and not item.startswith("_")
            ):
                return getattr(
                    [v for k, v in self.tuple_prior_tuples if item.split("_")[0] == k][
                        0
                    ],
                    item,
                )

        except IndexError:
            pass

        self.__getattribute__(item)

    @property
    def is_deferred_arguments(self):
        return len(self.direct_deferred_tuples) > 0

    # noinspection PyUnresolvedReferences
    @check_assertions
    def _instance_for_arguments(self, arguments: {ModelObject: object}):
        """
        Returns an instance of the associated class for a set of arguments

        Parameters
        ----------
        arguments: {Prior: float}
            Dictionary mapping_matrix priors to attribute analysis_path and value pairs

        Returns
        -------
            An instance of the class
        """
        model_arguments = dict()
        attribute_arguments = {
            key: value
            for key, value in self.__dict__.items()
            if key in self.constructor_argument_names
        }

        for tuple_prior in self.tuple_prior_tuples:
            model_arguments[tuple_prior.name] = tuple_prior.prior.value_for_arguments(
                arguments
            )
        for prior_model_tuple in self.direct_prior_model_tuples:
            prior_model = prior_model_tuple.prior_model
            model_arguments[
                prior_model_tuple.name
            ] = prior_model.instance_for_arguments(arguments,)

        prior_arguments = dict()

        for name, prior in self.direct_prior_tuples:
            try:
                prior_arguments[name] = arguments[prior]
            except KeyError as e:
                raise KeyError(f"No argument given for prior {name}") from e

        constructor_arguments = {
            **attribute_arguments,
            **model_arguments,
            **prior_arguments,
        }

        if self.is_deferred_arguments:
            return DeferredInstance(self.cls, constructor_arguments)

        if not inspect.isclass(self.cls):
            result = object.__new__(inspect._findclass(self.cls))
            cls = self.cls
            cls(result, **constructor_arguments)
        else:
            result = self.cls(**constructor_arguments)

        for key, value in self.__dict__.items():
            if (
                not hasattr(result, key)
                and not isinstance(value, Prior)
                and not key == "cls"
                and not key.startswith("_")
            ):
                if isinstance(value, Model):
                    value = value.instance_for_arguments(arguments)
                elif isinstance(value, Prior):
                    value = arguments[value]
                try:
                    setattr(result, key, value)
                except AttributeError:
                    pass

        return result

    def gaussian_prior_model_for_arguments(self, arguments):
        """
        Returns a new instance of model mapper with a set of Gaussian priors based on \
        tuples provided by a previous nonlinear search.

        Parameters
        ----------
        arguments: [(float, float)]
            Tuples providing the mean and sigma of gaussians

        Returns
        -------
        new_model: ModelMapper
            A new model mapper populated with Gaussian priors
        """
        self.unfreeze()
        new_model = copy.deepcopy(self)

        new_model._assertions = list()

        model_arguments = {t.name: arguments[t.prior] for t in self.direct_prior_tuples}

        for tuple_prior_tuple in self.tuple_prior_tuples:
            setattr(
                new_model,
                tuple_prior_tuple.name,
                tuple_prior_tuple.prior.gaussian_tuple_prior_for_arguments(arguments),
            )
        for prior_tuple in self.direct_prior_tuples:
            setattr(new_model, prior_tuple.name, model_arguments[prior_tuple.name])
        for instance_tuple in self.direct_instance_tuples:
            setattr(new_model, instance_tuple.name, instance_tuple.instance)

        for name, prior_model in self.direct_prior_model_tuples:
            setattr(
                new_model,
                name,
                prior_model.gaussian_prior_model_for_arguments(arguments),
            )

        return new_model
