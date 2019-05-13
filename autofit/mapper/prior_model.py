import copy
import inspect
import re

from typing_inspect import is_tuple_type

from autofit import conf, exc
from autofit.mapper.model import ModelInstance
from autofit.mapper.model_object import ModelObject
from autofit.mapper.prior import cast_collection, PriorNameValue, ConstantNameValue, TuplePrior, UniformPrior, \
    LogUniformPrior, GaussianPrior, Constant, Prior, AttributeNameValue
from autofit.tools import dimension_type


def tuple_name(attribute_name):
    """
    Extract the name of a tuple attribute from the name of one of its components, e.g. centre_0 -> origin

    Parameters
    ----------
    attribute_name: str
        The name of an attribute which is a component of a tuple

    Returns
    -------
    tuple_name: str
        The name of the tuple of which the attribute is a member
    """
    return "_".join(attribute_name.split("_")[:-1])


def is_tuple_like_attribute_name(attribute_name):
    """
    Determine if a string matches the pattern "{attribute_name}_#", that is if it seems to be a tuple.

    Parameters
    ----------
    attribute_name: str
        The name of some attribute that may refer to a tuple.

    Returns
    -------
    is_tuple_like: bool
        True iff the attribute name looks like that which refers to a tuple.
    """
    pattern = re.compile("^[a-zA-Z_0-9]*_[0-9]$")
    return pattern.match(attribute_name)


class PriorModelNameValue(AttributeNameValue):
    @property
    def prior_model(self):
        return self.value


class AbstractPriorModel(ModelObject):
    """
    Abstract model that maps a set of priors to a particular class. Must be overridden by any prior model so that the \
    model mapper recognises its prior model attributes.

    @DynamicAttrs
    """

    @property
    def name(self):
        return self.__class__.__name__

    @staticmethod
    def from_object(t, *args, **kwargs):
        if inspect.isclass(t):
            obj = object.__new__(PriorModel)
            obj.__init__(t, **kwargs)
        elif isinstance(t, list) or isinstance(t, dict):
            obj = object.__new__(CollectionPriorModel)
            obj.__init__(arguments=t)
        else:
            obj = t
        return obj

    @property
    def info(self):
        info = []

        prior_model_iterator = self.direct_prior_tuples + self.direct_constant_tuples

        for attribute_tuple in prior_model_iterator:
            attribute = attribute_tuple[1]

            line = attribute_tuple.name
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
        raise NotImplementedError("PriorModels must implement the flat_prior_models property")

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
        return [(name, value) for name, value in self.__dict__.items() if isinstance(value, AbstractPriorModel)]

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
        return list(filter(lambda t: t[0] != "id" and isinstance(t[1], class_type), self.__dict__.items()))


def prior_for_class_and_attribute_name(cls, attribute_name):
    config_arr = conf.instance.prior_default.get_for_nearest_ancestor(cls, attribute_name)
    if config_arr[0] == "u":
        return UniformPrior(config_arr[1], config_arr[2])
    elif config_arr[0] == "l":
        return LogUniformPrior(config_arr[1], config_arr[2])
    elif config_arr[0] == "g":
        limits = conf.instance.prior_limit.get_for_nearest_ancestor(cls, attribute_name)
        return GaussianPrior(config_arr[1], config_arr[2], *limits)
    elif config_arr[0] == "c":
        return Constant(config_arr[1])
    raise exc.PriorException(
        "Default prior for {} has no type indicator (u - Uniform, g - Gaussian, c - Constant".format(
            attribute_name))


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
            defaults = dict(zip(arg_spec.args[-len(arg_spec.defaults):], arg_spec.defaults))
        except TypeError:
            defaults = {}

        args = arg_spec.args[1:]

        if 'settings' in defaults:
            del defaults['settings']
        if 'settings' in args:
            args.remove('settings')

        for arg in args:
            if isinstance(defaults.get(arg), str):
                continue

            if arg in kwargs:
                ls = CollectionPriorModel([])
                for obj in kwargs[arg]:
                    if inspect.isclass(obj):
                        ls.append(AbstractPriorModel.from_object(obj))
                    else:
                        ls.append(obj)

                setattr(self, arg, ls)
            elif arg in defaults and isinstance(defaults[arg], tuple):
                tuple_prior = TuplePrior()
                for i in range(len(defaults[arg])):
                    attribute_name = "{}_{}".format(arg, i)
                    setattr(tuple_prior, attribute_name, self.make_prior(attribute_name))
                setattr(self, arg, tuple_prior)
            elif arg in arg_spec.annotations and arg_spec.annotations[arg] != float:
                spec = arg_spec.annotations[arg]
                # noinspection PyUnresolvedReferences
                if issubclass(spec, float):
                    setattr(self, arg, AnnotationPriorModel(spec, cls, arg))
                elif is_tuple_type(spec):
                    tuple_prior = TuplePrior()
                    for i, tuple_arg in enumerate(spec.__args__):
                        attribute_name = "{}_{}".format(arg, i)
                        setattr(tuple_prior, attribute_name, self.make_prior(attribute_name))
                    setattr(self, arg, tuple_prior)
                else:
                    setattr(self, arg, PriorModel(arg_spec.annotations[arg]))
            else:
                setattr(self, arg, self.make_prior(arg))

    def __eq__(self, other):
        return isinstance(other, PriorModel) and self.cls == other.cls and self.prior_tuples == other.prior_tuples

    def make_prior(self, attribute_name):
        """
        Create a prior for an attribute of a class with a given name. The prior is created by searching the default
        prior config for the attribute.

        Entries in configuration with a u become uniform priors; with a g become gaussian priors; with a c become
        constants.

        If prior configuration for a given attribute is not specified in the configuration for a class then the
        configuration corresponding to the parents of that class is searched. If no configuration can be found then a
        prior exception is raised.

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
        return prior_for_class_and_attribute_name(self.cls, attribute_name)

    def linked_model_for_class(self, cls, make_constants_variable=False, **kwargs):
        """
        Create a PriorModel wrapping the specified class with attributes from this instance. Priors can be overridden
        using keyword arguments. Any constructor arguments of the new class for which there is no attribute associated
        with this class and no keyword argument are created from config.

        If make_constants_variable is True then constants associated with this instance will be used to set the mean
        of priors in the new instance rather than overriding them.

        Parameters
        ----------
        cls: class
            The class that the new PriorModel will wrap
        make_constants_variable: bool
            If True constants from this instance will be used to determine the mean values for priors in the new
            instance rather than overriding them
        kwargs
            Keyword arguments passed in here are used to override attributes from this instance or add new attributes

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
                    is_tuple_like_attribute_name(name) and tuple_name(name) in constructor_args):
                attribute = kwargs[name] if name in kwargs else attribute_tuple.value
                if make_constants_variable and isinstance(attribute, Constant):
                    new_attribute = getattr(new_model, name)
                    if isinstance(new_attribute, Prior):
                        new_attribute.mean = attribute.value
                        continue
                setattr(new_model, name, attribute)
        return new_model

    def __setattr__(self, key, value):
        if key not in ("component_number", "phase_property_position", "mapping_name", "id"):
            try:
                if "_" in key:
                    name = key.split("_")[0]
                    tuple_prior = [v for k, v in self.tuple_prior_tuples if name == k][0]
                    setattr(tuple_prior, key, value)
                    return
            except IndexError:
                pass
            if isinstance(value, float) or isinstance(value, int):
                super().__setattr__(key, Constant(value))
                return
        super(PriorModel, self).__setattr__(key, value)

    def __getattr__(self, item):
        try:
            if "_" in item:
                return getattr([v for k, v in self.tuple_prior_tuples if item.split("_")[0] == k][0], item)

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
    @cast_collection(ConstantNameValue)
    def direct_constant_tuples(self):
        """
        Returns
        -------
        constants: [(String, Constant)]
            A list of constants
        """
        return self.tuples_with_type(Constant)

    @property
    @cast_collection(ConstantNameValue)
    def constant_tuples(self):
        """
        Returns
        -------
        constants: [(String, Constant)]
        """
        return [constant_tuple for tuple_prior in self.tuple_prior_tuples for constant_tuple in
                tuple_prior[1].constant_tuples] + self.direct_constant_tuples

    @property
    @cast_collection(AttributeNameValue)
    def attribute_tuples(self):
        return self.prior_tuples + self.constant_tuples

    @property
    def prior_class_dict(self):
        return {prior[1]: self.cls for prior in self.prior_tuples}

    def instance_for_arguments(self, arguments: {Prior: float}):
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
            prior.assert_within_limits(value)
        model_arguments = {t.name: arguments[t.prior] for t in self.direct_prior_tuples}
        constant_arguments = {t.name: t.constant.value for t in self.direct_constant_tuples}
        for tuple_prior in self.tuple_prior_tuples:
            model_arguments[tuple_prior.name] = tuple_prior.prior.value_for_arguments(arguments)
        for prior_model_tuple in self.direct_prior_model_tuples:
            model_arguments[prior_model_tuple.name] = prior_model_tuple.prior_model.instance_for_arguments(arguments)

        return self.cls(**{**model_arguments, **constant_arguments})

    def gaussian_prior_model_for_arguments(self, arguments):
        """
        Create a new instance of model mapper with a set of Gaussian priors based on tuples provided by a previous \
        nonlinear search.

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
                    tuple_prior_tuple.prior.gaussian_tuple_prior_for_arguments(arguments))
        for prior_tuple in self.direct_prior_tuples:
            setattr(new_model, prior_tuple.name, model_arguments[prior_tuple.name])
        for constant_tuple in self.constant_tuples:
            setattr(new_model, constant_tuple.name, constant_tuple.constant)

        for name, prior_model in self.direct_prior_model_tuples:
            setattr(new_model, name, prior_model.gaussian_prior_model_for_arguments(arguments))

        return new_model


class AnnotationPriorModel(PriorModel):
    def __init__(self, cls, parent_class, true_argument_name, **kwargs):
        self.parent_class = parent_class
        self.true_argument_name = true_argument_name
        super().__init__(cls, **kwargs)

    def make_prior(self, attribute_name):
        return prior_for_class_and_attribute_name(self.parent_class, self.true_argument_name)


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
        return [value for key, value in self.__dict__.items() if key not in ('component_number', 'item_number', 'id')]

    @property
    def flat_prior_model_tuples(self):
        return [flat_prior_model for prior_model in self.prior_models for flat_prior_model in
                prior_model.flat_prior_model_tuples]

    def __init__(self, arguments=None):
        """
        A prior model used to represent a list of prior models for convenience.

        Parameters
        ----------
        arguments: list
            A list classes, prior_models or instances
        """
        super().__init__()
        self.component_number = next(self._ids)

        self.item_number = 0

        if isinstance(arguments, list):
            for argument in arguments:
                self.append(argument)
        if isinstance(arguments, dict):
            for key, value in arguments.items():
                setattr(self, key, AbstractPriorModel.from_object(value))

    def __add__(self, other):
        new = CollectionPriorModel()
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
        return [(prior_model.mapping_name if hasattr(prior_model, "mapping_name") else str(i), prior_model) for
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
                for key, value in self.__dict__.items() if key not in ('component_number', 'item_number', 'id')
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
        return set([prior for prior_model in self.prior_models for prior in prior_model.prior_tuples])

    @property
    @cast_collection(ConstantNameValue)
    def constant_tuples(self):
        """
        Returns
        -------
        priors: [(String, Union(Prior, TuplePrior))]
        """
        return set([constant for prior_model in self.prior_models for constant in prior_model.constant_tuples])

    @property
    def prior_class_dict(self):
        return {prior: cls for prior_model in self.prior_models for prior, cls in prior_model.prior_class_dict.items()}
