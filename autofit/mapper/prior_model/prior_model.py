import copy
import inspect
import logging

from typing_inspect import is_tuple_type

from autofit import exc
from autofit.mapper.model_object import ModelObject
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.deferred import DeferredInstance
from autofit.mapper.prior_model.prior import TuplePrior, Prior
from autofit.tools.promise import Promise

logger = logging.getLogger(__name__)


class PriorModel(AbstractPriorModel):
    """Object comprising class and associated priors
        @DynamicAttrs
    """

    @property
    def name(self):
        return self.cls.__name__

    def as_model(self):
        return PriorModel(self.cls)

    def __hash__(self):
        return self.id

    def __add__(self, other):
        if self.cls != other.cls:
            raise TypeError(
                f"Cannot add PriorModels with different classes "
                f"({self.cls.__name__} and {other.cls.__name__})"
            )
        return super().__add__(other)

    @property
    def constructor_argument_names(self):
        return inspect.getfullargspec(self.cls.__init__).args[1:]

    def __init__(self, cls, **kwargs):
        """
        Parameters
        ----------
        cls: class
            The class associated with this instance
        """
        super().__init__()
        if cls is self:
            return

        self.cls = cls

        arg_spec = inspect.getfullargspec(cls.__init__)

        try:
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
                    from autofit.mapper.prior_model.collection import (
                        CollectionPriorModel,
                    )

                    ls = CollectionPriorModel(keyword_arg)

                    setattr(self, arg, ls)
                else:
                    if inspect.isclass(keyword_arg):
                        keyword_arg = PriorModel(keyword_arg)
                    setattr(self, arg, keyword_arg)
            elif arg in defaults and isinstance(defaults[arg], tuple):
                tuple_prior = TuplePrior()
                for i in range(len(defaults[arg])):
                    attribute_name = "{}_{}".format(arg, i)
                    setattr(
                        tuple_prior, attribute_name, self.make_prior(attribute_name)
                    )
                setattr(self, arg, tuple_prior)
            elif arg in arg_spec.annotations and arg_spec.annotations[arg] != float:
                spec = arg_spec.annotations[arg]
                # noinspection PyUnresolvedReferences
                if inspect.isclass(spec) and issubclass(spec, float):
                    from autofit.mapper.prior_model.annotation import (
                        AnnotationPriorModel,
                    )

                    setattr(self, arg, AnnotationPriorModel(spec, cls, arg))
                elif is_tuple_type(spec):
                    tuple_prior = TuplePrior()
                    for i, tuple_arg in enumerate(spec.__args__):
                        attribute_name = "{}_{}".format(arg, i)
                        setattr(
                            tuple_prior, attribute_name, self.make_prior(attribute_name)
                        )
                    setattr(self, arg, tuple_prior)
                else:
                    setattr(self, arg, PriorModel(arg_spec.annotations[arg]))
            else:
                setattr(self, arg, self.make_prior(arg))
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(
                    self, key, PriorModel(value) if inspect.isclass(value) else value
                )

    def __eq__(self, other):
        return (
            isinstance(other, PriorModel)
            and self.cls == other.cls
            and self.prior_tuples == other.prior_tuples
        )

    def make_prior(self, attribute_name):
        """
        Create a prior for an attribute of a class with a given name. The prior is
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
        return Prior.for_class_and_attribute_name(self.cls, attribute_name)

    def __setattr__(self, key, value):
        if key not in (
            "component_number",
            "phase_property_position",
            "mapping_name",
            "id",
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
            if "_" in item:
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
    def prior_class_dict(self):
        from autofit.mapper.prior_model.annotation import AnnotationPriorModel

        d = {prior[1]: self.cls for prior in self.prior_tuples}
        for prior_model in self.prior_model_tuples:
            if not isinstance(prior_model[1], AnnotationPriorModel):
                d.update(prior_model[1].prior_class_dict)
        return d

    @property
    def is_deferred_arguments(self):
        return len(self.direct_deferred_tuples) > 0

    # noinspection PyUnresolvedReferences
    def instance_for_arguments(self, arguments: {ModelObject: object}):
        """
        Create an instance of the associated class for a set of arguments

        Parameters
        ----------
        arguments: {Prior: float}
            Dictionary mapping_matrix priors to attribute analysis_path and value pairs

        Returns
        -------
            An instance of the class
        """
        if self.promise_count > 0:
            raise exc.PriorException(
                "All promises must be populated prior to instantiation"
            )
        for prior, value in arguments.items():
            if isinstance(value, float) or isinstance(value, int):
                prior.assert_within_limits(value)

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
            ] = prior_model.instance_for_arguments(arguments)

        constructor_arguments = {
            **attribute_arguments,
            **model_arguments,
            **{t.name: arguments[t.prior] for t in self.direct_prior_tuples},
        }

        if self.is_deferred_arguments:
            return DeferredInstance(self.cls, constructor_arguments)

        try:
            result = self.cls(**constructor_arguments)
        except TypeError as e:
            raise e

        for key, value in self.__dict__.items():
            if (
                not hasattr(result, key)
                and not isinstance(value, Prior)
                and not isinstance(value, Promise)
            ):
                if isinstance(value, PriorModel):
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
        Create a new instance of model mapper with a set of Gaussian priors based on \
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
        new_model = copy.deepcopy(self)

        model_arguments = {t.name: arguments[t.prior] for t in self.direct_prior_tuples}

        for tuple_prior_tuple in self.tuple_prior_tuples:
            setattr(
                new_model,
                tuple_prior_tuple.name,
                tuple_prior_tuple.prior.gaussian_tuple_prior_for_arguments(arguments),
            )
        for prior_tuple in self.direct_prior_tuples:
            setattr(new_model, prior_tuple.name, model_arguments[prior_tuple.name])
        for instance_tuple in self.instance_tuples:
            setattr(new_model, instance_tuple.name, instance_tuple.instance)

        for name, prior_model in self.direct_prior_model_tuples:
            setattr(
                new_model,
                name,
                prior_model.gaussian_prior_model_for_arguments(arguments),
            )

        return new_model
