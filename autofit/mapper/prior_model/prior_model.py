import copy
import inspect

from typing_inspect import is_tuple_type

from autofit.mapper.model_object import ModelObject
from autofit.mapper.prior import cast_collection, PriorNameValue, ConstantNameValue, \
    TuplePrior, Prior, AttributeNameValue, \
    DeferredNameValue
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.deferred import DeferredArgument
from autofit.mapper.prior_model.deferred import DeferredInstance
from autofit.mapper.prior_model.util import tuple_name, is_tuple_like_attribute_name


class PriorModel(AbstractPriorModel):
    """Object comprising class and associated priors
        @DynamicAttrs
    """

    @property
    def name(self):
        return self.cls.__name__

    @property
    def flat_prior_model_tuples(self):
        return [("", self)]

    def __hash__(self):
        return self.id

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
        self.component_number = next(self._ids)

        arg_spec = inspect.getfullargspec(cls.__init__)

        try:
            defaults = dict(
                zip(arg_spec.args[-len(arg_spec.defaults):], arg_spec.defaults))
        except TypeError:
            defaults = {}

        args = self.constructor_argument_names

        if 'settings' in defaults:
            del defaults['settings']
        if 'settings' in args:
            args.remove('settings')

        for arg in args:
            if isinstance(defaults.get(arg), str):
                continue

            if arg in kwargs:
                keyword_arg = kwargs[arg]
                if isinstance(keyword_arg, list):
                    from autofit.mapper.prior_model.collection import \
                        CollectionPriorModel
                    ls = CollectionPriorModel([])
                    for obj in keyword_arg:
                        if inspect.isclass(obj):
                            ls.append(AbstractPriorModel.from_object(obj))
                        else:
                            ls.append(obj)
                    setattr(self, arg, ls)
                else:
                    setattr(self, arg, keyword_arg)
            elif arg in defaults and isinstance(defaults[arg], tuple):
                tuple_prior = TuplePrior()
                for i in range(len(defaults[arg])):
                    attribute_name = "{}_{}".format(arg, i)
                    setattr(tuple_prior, attribute_name,
                            self.make_prior(attribute_name))
                setattr(self, arg, tuple_prior)
            elif arg in arg_spec.annotations and arg_spec.annotations[arg] != float:
                spec = arg_spec.annotations[arg]
                # noinspection PyUnresolvedReferences
                if issubclass(spec, float):
                    from autofit.mapper.prior_model.annotation import \
                        AnnotationPriorModel
                    setattr(self, arg, AnnotationPriorModel(spec, cls, arg))
                elif is_tuple_type(spec):
                    tuple_prior = TuplePrior()
                    for i, tuple_arg in enumerate(spec.__args__):
                        attribute_name = "{}_{}".format(arg, i)
                        setattr(tuple_prior, attribute_name,
                                self.make_prior(attribute_name))
                    setattr(self, arg, tuple_prior)
                else:
                    setattr(self, arg, PriorModel(arg_spec.annotations[arg]))
            else:
                setattr(self, arg, self.make_prior(arg))

    def __eq__(self, other):
        return isinstance(
            other,
            PriorModel) \
               and self.cls == other.cls \
               and self.prior_tuples == other.prior_tuples

    def make_prior(self, attribute_name):
        """
        Create a prior for an attribute of a class with a given name. The prior is
        created by searching the default prior config for the attribute.

        Entries in configuration with a u become uniform priors; with a g become
        gaussian priors; with a c become constants.

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
        return Prior.for_class_and_attribute_name(
            self.cls,
            attribute_name
        )

    def linked_model_for_class(self, cls, make_constants_variable=False, **kwargs):
        """
        Create a PriorModel wrapping the specified class with attributes from this
        instance. Priors can be overridden using keyword arguments. Any constructor
        arguments of the new class for which there is no attribute associated with
        this class and no keyword argument are created from config.

        If make_constants_variable is True then constants associated with this
        instance will be used to set the mean of priors in the new instance rather
        than overriding them.

        Parameters
        ----------
        cls: class
            The class that the new PriorModel will wrap
        make_constants_variable: bool
            If True constants from this instance will be used to determine the mean
            values for priors in the new instance rather than overriding them
        kwargs
            Keyword arguments passed in here are used to override attributes from this
            instance or add new attributes

        Returns
        -------
        new_model: PriorModel
            A new prior model with priors derived from this instance
        """
        constructor_args = inspect.getfullargspec(cls).args
        attribute_tuples = self.attribute_tuples
        new_model = PriorModel(cls)
        for attribute_tuple in attribute_tuples:
            name = attribute_tuple.name
            if name in constructor_args or (
                    is_tuple_like_attribute_name(
                        name
                    ) and tuple_name(name) in constructor_args):
                attribute = kwargs[name] if name in kwargs else attribute_tuple.value
                if make_constants_variable and (isinstance(
                        attribute,
                        float
                ) or isinstance(
                    attribute,
                    int
                )):
                    new_attribute = getattr(new_model, name)
                    if isinstance(new_attribute, Prior):
                        new_attribute.mean = attribute
                        continue
                setattr(new_model, name, attribute)
        return new_model

    def __setattr__(self, key, value):
        if key not in (
                "component_number", "phase_property_position", "mapping_name", "id"):
            try:
                if "_" in key:
                    name = key.split("_")[0]
                    tuple_prior = [v for k, v in self.tuple_prior_tuples if name == k][
                        0]
                    setattr(tuple_prior, key, value)
                    return
            except IndexError:
                pass
        super(PriorModel, self).__setattr__(key, value)

    def __getattr__(self, item):
        try:
            if "_" in item:
                return getattr(
                    [v for k, v in self.tuple_prior_tuples if item.split("_")[0] == k][
                        0], item)

        except IndexError:
            pass
        self.__getattribute__(item)

    @property
    @cast_collection(PriorNameValue)
    def tuple_prior_tuples(self):
        """
        Returns
        -------
        tuple_prior_tuples: [(String, TuplePrior)]
        """
        return self.tuples_with_type(TuplePrior)

    @property
    @cast_collection(PriorNameValue)
    def direct_prior_tuples(self):
        """
        Returns
        -------
        direct_priors: [(String, Prior)]
        """
        return self.tuples_with_type(Prior)

    @property
    @cast_collection(PriorNameValue)
    def prior_tuples(self):
        """
        Returns
        -------
        priors: [(String, Prior))]
        """
        # noinspection PyUnresolvedReferences
        deeper = [
            (prior_model[0] if prior.name == "value" else prior.name, prior.value)
            for prior_model in
            self.prior_model_tuples
            for prior in
            prior_model[1].prior_tuples]
        tuple_priors = [prior for tuple_prior in self.tuple_prior_tuples for prior in
                        tuple_prior[1].prior_tuples]
        direct_priors = self.direct_prior_tuples
        return tuple_priors + direct_priors + deeper

    @property
    @cast_collection(DeferredNameValue)
    def direct_deferred_tuples(self):
        return self.tuples_with_type(DeferredArgument)

    @property
    @cast_collection(ConstantNameValue)
    def constant_tuples(self):
        """
        Returns
        -------
        constants: [(String, Constant)]
        """
        return [constant_tuple for tuple_prior in self.tuple_prior_tuples for
                constant_tuple in
                tuple_prior[1].constant_tuples] + self.direct_constant_tuples

    @property
    @cast_collection(AttributeNameValue)
    def attribute_tuples(self):
        return self.prior_tuples + self.constant_tuples

    @property
    def prior_class_dict(self):
        return {prior[1]: self.cls for prior in self.prior_tuples}

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
        for prior, value in arguments.items():
            if isinstance(value, float) or isinstance(value, int):
                prior.assert_within_limits(value)

        model_arguments = {
            t.name: arguments[t.prior]
            for t in self.direct_prior_tuples
        }
        attribute_arguments = {
            key: value for key, value in self.__dict__.items()
            if key in self.constructor_argument_names
        }

        for tuple_prior in self.tuple_prior_tuples:
            model_arguments[tuple_prior.name] = tuple_prior.prior.value_for_arguments(
                arguments)
        for prior_model_tuple in self.direct_prior_model_tuples:
            prior_model = prior_model_tuple.prior_model
            model_arguments[
                prior_model_tuple.name] = prior_model.instance_for_arguments(
                arguments
            )

        constructor_arguments = {
            **attribute_arguments,
            **model_arguments
        }

        if self.is_deferred_arguments:
            return DeferredInstance(self.cls, constructor_arguments)
        return self.cls(**constructor_arguments)

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
            setattr(new_model, tuple_prior_tuple.name,
                    tuple_prior_tuple.prior.gaussian_tuple_prior_for_arguments(
                        arguments))
        for prior_tuple in self.direct_prior_tuples:
            setattr(new_model, prior_tuple.name, model_arguments[prior_tuple.name])
        for constant_tuple in self.constant_tuples:
            setattr(new_model, constant_tuple.name, constant_tuple.constant)

        for name, prior_model in self.direct_prior_model_tuples:
            setattr(new_model, name,
                    prior_model.gaussian_prior_model_for_arguments(arguments))

        return new_model
