import copy
import inspect
from functools import wraps
from numbers import Number
from random import random
from typing import Tuple, Optional

import numpy as np

from autofit import exc
from autofit.mapper import model
from autofit.mapper import model_mapper
from autofit.mapper.model import AbstractModel
from autofit.mapper.prior.deferred import DeferredArgument
from autofit.mapper.prior.prior import GaussianPrior
from autofit.mapper.prior.prior import TuplePrior, Prior, WidthModifier, Limits
from autofit.mapper.prior_model import collection
from autofit.mapper.prior_model import dimension_type as dim
from autofit.mapper.prior_model.attribute_pair import DeferredNameValue
from autofit.mapper.prior_model.attribute_pair import cast_collection, PriorNameValue, InstanceNameValue
from autofit.mapper.prior_model.recursion import DynamicRecursionCache
from autofit.mapper.prior_model.util import PriorModelNameValue
from autofit.text.formatter import TextFormatter


def check_assertions(func):
    @wraps(func)
    def wrapper(s, arguments):
        # noinspection PyProtectedMember
        failed_assertions = [
            assertion
            for assertion
            in s._assertions
            if assertion is False or assertion is not True and not assertion.instance_for_arguments(
                arguments
            )
        ]
        number_of_failed_assertions = len(failed_assertions)
        if number_of_failed_assertions > 0:
            name_string = "\n".join([
                assertion.name
                for assertion
                in failed_assertions
                if hasattr(assertion, "name") and assertion.name is not None
            ])
            raise exc.FitException(
                f"{number_of_failed_assertions} assertions failed!\n{name_string}"
            )

        return func(s, arguments)

    return wrapper


class AbstractPriorModel(AbstractModel):
    """
    Abstract model that maps a set of priors to a particular class. Must be
    overridden by any prior model so that the model mapper recognises its prior \
    model attributes.
    @DynamicAttrs
    """

    def __init__(self):
        super().__init__()
        self._assertions = list()

    def add_assertion(self, assertion, name=None):
        """
        Assert that some relationship holds between physical values associated with
        priors at the point an instance is created. If this fails a FitException is
        raised causing the model to be re-sampled.
        Parameters
        ----------
        assertion
            An assertion that one prior must be greater than another.
        name
            A name describing the assertion that is logged when it is violated.
        """
        if assertion is True:
            return
        try:
            assertion.name = name
        except AttributeError:
            pass
        self._assertions.append(assertion)

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
            obj = object.__new__(
                collection.CollectionPriorModel
            )
            obj.__init__(t)
        else:
            obj = t
        return obj

    def instance_from_unit_vector(self, unit_vector, assert_priors_in_limits=True):
        """
        Creates a ModelInstance, which has an attribute and class instance corresponding
        to every PriorModel attributed to this instance.
        This method takes as input a unit vector of parameter values, converting each to
        physical values via their priors.
        Parameters
        ----------
        unit_vector: [float]
            A unit hypercube vector that is mapped to an instance of physical values via the priors.
        Returns
        -------
        model_instance : autofit.mapper.model.ModelInstance
            An object containing reconstructed model_mapper instances
        Raises
        ------
        exc.FitException
            If any assertion attached to this object returns False.
        """
        arguments = dict(
            map(
                lambda prior_tuple, unit: (
                    prior_tuple.prior,
                    prior_tuple.prior.value_for(unit),
                ),
                self.prior_tuples_ordered_by_id,
                unit_vector,
            )
        )

        return self.instance_for_arguments(arguments, assert_priors_in_limits=assert_priors_in_limits)

    @property
    @cast_collection(PriorNameValue)
    def unique_prior_tuples(self):
        """
        Returns
        -------
        prior_tuple_dict: [(Prior, PriorTuple)]
            The set of all priors associated with this mapper
        """
        return {
            prior_tuple[1]: prior_tuple
            for prior_tuple in self.attribute_tuples_with_type(Prior)
        }.values()

    @property
    def unique_promise_tuples(self):
        from autofit.mapper.prior.promise import AbstractPromise

        return {
            prior_tuple[1]: prior_tuple
            for prior_tuple in self.attribute_tuples_with_type(AbstractPromise)
        }.values()

    @property
    @cast_collection(PriorNameValue)
    def prior_tuples_ordered_by_id(self):
        """
        Returns
        -------
        priors: [Prior]
            An ordered list of unique priors associated with this mapper
        """
        return sorted(
            list(self.unique_prior_tuples), key=lambda prior_tuple: prior_tuple.prior.id
        )

    def vector_from_unit_vector(self, unit_vector):
        """
        Parameters
        ----------
        unit_vector: [float]
            A unit hypercube vector
        Returns
        -------
        values: [float]
            A vector with values output by priors
        """
        return list(
            map(
                lambda prior_tuple, unit: prior_tuple.prior.value_for(unit),
                self.prior_tuples_ordered_by_id,
                unit_vector,
            )
        )

    def random_unit_vector_within_limits(self, lower_limit=0.0, upper_limit=1.0):
        """ Generate a random vector of unit values by drawing uniform random values between 0 and 1.
        Returns
        -------
        unit_values: [float]
            A list of unit values constructed by taking random values from each prior.
        """
        return list(np.random.uniform(low=lower_limit, high=upper_limit, size=self.prior_count))

    def random_vector_from_priors_within_limits(self, lower_limit, upper_limit):
        """ Generate a random vector of physical values by drawing uniform random values between an input lower and
        upper limit and using the model priors to map them from unit values to physical values.
        This is used for MCMC initialization, whereby the starting points of a walker(s) is confined to a restricted
        range of prior space. In particular, it is used for generate the "ball" initialization of Emcee.
        Returns
        -------
        physical_values: [float]
            A list of physical values constructed by taking the mean possible value from
            each prior.
        """

        while True:

            vector = self.vector_from_unit_vector(
                list(np.random.uniform(low=lower_limit, high=upper_limit, size=self.prior_count))
            )

            try:
                self.instance_from_vector(vector=vector)
                return vector
            except exc.PriorLimitException:
                pass

    @property
    def random_vector_from_priors(self):
        """ Generate a random vector of physical values by drawing uniform random values between 0 and 1 and using
        the model priors to map them from unit values to physical values.
        Returns
        -------
        physical_values: [float]
            A list of physical values constructed by taking random values from each prior.
        """
        return self.random_vector_from_priors_within_limits(lower_limit=0.0, upper_limit=1.0)

    @property
    def physical_values_from_prior_medians(self):
        """
        Returns
        -------
        physical_values: [float]
            A list of physical values constructed by taking the mean possible value from
            each prior.
        """
        return self.vector_from_unit_vector([0.5] * len(self.unique_prior_tuples))

    def instance_from_vector(
            self,
            vector,
            assert_priors_in_limits=True
    ):
        """
        Creates a ModelInstance, which has an attribute and class instance corresponding
        to every PriorModel attributed to this instance.
        This method takes as input a physical vector of parameter values, thus omitting
        the use of priors.
        Parameters
        ----------
        vector: [float]
            A vector of physical parameter values that is mapped to an instance.
        assert_priors_in_limits
            If True it is checked that the physical values of priors are within set limits
        Returns
        -------
        model_instance : autofit.mapper.model.ModelInstance
            An object containing reconstructed model_mapper instances
        """
        arguments = dict(
            map(
                lambda prior_tuple, physical_unit: (prior_tuple.prior, physical_unit),
                self.prior_tuples_ordered_by_id,
                vector,
            )
        )

        return self.instance_for_arguments(
            arguments,
            assert_priors_in_limits=assert_priors_in_limits
        )

    def mapper_from_partial_prior_arguments(self, arguments):
        """
        Creates a new model mapper from a dictionary mapping_matrix existing priors to
        new priors, keeping existing priors where no mapping is provided.
        Parameters
        ----------
        arguments: {Prior: Prior}
            A dictionary mapping_matrix priors to priors
        Returns
        -------
        model_mapper: ModelMapper
            A new model mapper with updated priors.
        """
        original_prior_dict = {prior: prior for prior in self.priors}
        return self.mapper_from_prior_arguments({**original_prior_dict, **arguments})

    def mapper_from_prior_arguments(self, arguments):
        """
        Creates a new model mapper from a dictionary mapping_matrix existing priors to
        new priors.
        Parameters
        ----------
        arguments: {Prior: Prior}
            A dictionary mapping_matrix priors to priors
        Returns
        -------
        model_mapper: ModelMapper
            A new model mapper with updated priors.
        """
        mapper = copy.deepcopy(self)

        for prior_model_tuple in self.prior_model_tuples:
            setattr(
                mapper,
                prior_model_tuple.name,
                prior_model_tuple.prior_model.gaussian_prior_model_for_arguments(
                    arguments
                ),
            )

        return mapper

    def mapper_from_gaussian_tuples(
            self,
            tuples,
            a=None,
            r=None,
            use_errors=True,
            use_widths=True,
            no_limits=False
    ):
        """
        Creates a new model mapper from a list of floats describing the mean values
        of gaussian priors. The widths of the new priors are taken from the
        width_config. The new gaussian priors must be provided in the same order as
        the priors associated with model.
        If a is not None then all priors are created with an absolute width of a.
        If r is not None then all priors are created with a relative width of r.
        Parameters
        ----------
        no_limits
            If True generated priors have infinite limits
        r
            The relative width to be assigned to gaussian priors
        a
            print(tuples[i][1], width)
            The absolute width to be assigned to gaussian priors
        use_errors : bool
            If True, the passed errors of the model components estimated in a previous non-linear search (computed
            at the prior_passer.sigma value) are used to set the pass Gaussian Prior sigma value (if both width and
            passed errors are used, the maximum of these two values are used).
        use_widths : bool
            If True, the minimum prior widths specified in the json_prior configs of the model components are used to
            set the passed Gaussian Prior sigma value (if both widths and passed errors are used, the maximum of
            these two values are used).
        tuples
            A list of tuples each containing the mean and width of a prior
        Returns
        -------
        mapper: ModelMapper
            A new model mapper with all priors replaced by gaussian priors.
        """

        prior_tuples = self.prior_tuples_ordered_by_id
        prior_class_dict = self.prior_class_dict
        arguments = {}

        for i, prior_tuple in enumerate(prior_tuples):
            prior = prior_tuple.prior
            cls = prior_class_dict[prior]
            mean, sigma = tuples[i]

            name = prior_tuple.name
            # Use the name of the collection for configuration when a prior's name
            # is just a number (i.e. its position in a collection)
            if name.isdigit():
                name = self.path_for_prior(prior_tuple.prior)[-2]

            width_modifier = WidthModifier.for_class_and_attribute_name(cls, name)

            if a is not None and r is not None:
                raise exc.PriorException(
                    "Width of new priors cannot be both relative and absolute."
                )
            if a is not None:
                width = a
            elif r is not None:
                width = r * mean
            else:
                width = width_modifier(mean)

            if no_limits:
                limits = (float("-inf"), float("inf"))
            else:
                try:
                    limits = Limits.for_class_and_attributes_name(
                        cls,
                        name
                    )
                except exc.PriorException:
                    limits = prior.limits

            if use_errors and not use_widths:
                sigma = tuples[i][1]
            elif not use_errors and use_widths:
                sigma = width
            elif use_errors and use_widths:
                sigma = max(tuples[i][1], width)
            else:
                raise exc.PriorException("use_passed_errors and use_widths are both False, meeaning there is no "
                                         "way to pass priors to set up the new model's Gaussian Priors.")

            arguments[prior] = GaussianPrior(
                mean,
                sigma,
                *limits
            )

        return self.mapper_from_prior_arguments(arguments)

    def instance_from_prior_medians(self):
        """
        Creates a list of physical values from the median values of the priors.
        Returns
        -------
        physical_values : [float]
            A list of physical values
        """
        return self.instance_from_unit_vector(
            unit_vector=[0.5] * len(self.prior_tuples)
        )

    def log_priors_from_vector(
            self,
            vector: [float],
    ):
        """
        Compute the log priors of every parameter in a vector, using the Prior of every parameter.
        The log prior values are used by Emcee to map the log likelihood to the poserior of the model.
        Parameters
        ----------
        vector : [float]
            A vector of physical parameter values.
        Returns
        -------
        log_priors : []
            An list of the log prior value of every parameter.
        """
        return list(
            map(
                lambda prior_tuple, value: prior_tuple.prior.log_prior_from_value(value=value),
                self.prior_tuples_ordered_by_id,
                vector,
            )
        )

    def random_instance(self):
        """
        Creates a random instance of the model.
        """
        return self.instance_from_unit_vector(
            unit_vector=[random() for _ in self.prior_tuples]
        )

    @staticmethod
    @DynamicRecursionCache()
    def from_instance(instance, model_classes=tuple()):
        """
        Recursively create an prior object model from an object model.
        Parameters
        ----------
        model_classes
        instance
            A dictionary, list, class instance or model instance
        Returns
        -------
        abstract_prior_model
            A concrete child of an abstract prior model
        """

        if isinstance(instance, list):
            result = collection.CollectionPriorModel(
                [
                    AbstractPriorModel.from_instance(item, model_classes=model_classes)
                    for item in instance
                ]
            )
        elif isinstance(instance, model.ModelInstance):
            result = model_mapper.ModelMapper()
            for key, value in instance.dict.items():
                setattr(
                    result,
                    key,
                    AbstractPriorModel.from_instance(
                        value, model_classes=model_classes
                    ),
                )
        elif isinstance(instance, dict):
            result = collection.CollectionPriorModel(
                {
                    key: AbstractPriorModel.from_instance(
                        value, model_classes=model_classes
                    )
                    for key, value in instance.items()
                }
            )
        elif isinstance(instance, (dim.DimensionType, np.ndarray)):
            return instance
        else:
            from .prior_model import PriorModel

            try:
                result = PriorModel(
                    instance.__class__,
                    **{
                        key: AbstractPriorModel.from_instance(
                            value, model_classes=model_classes
                        )
                        for key, value in instance.__dict__.items()
                        if key != "cls"
                    },
                )
            except AttributeError:
                return instance
        if any([isinstance(instance, cls) for cls in model_classes]):
            return result.as_model()
        return result

    @property
    @cast_collection(PriorNameValue)
    def direct_prior_tuples(self):
        return self.direct_tuples_with_type(Prior)

    @property
    @cast_collection(InstanceNameValue)
    def direct_instance_tuples(self):
        return self.direct_tuples_with_type(float)

    @property
    @cast_collection(PriorModelNameValue)
    def prior_model_tuples(self):
        return self.direct_tuples_with_type(AbstractPriorModel)

    @property
    @cast_collection(PriorNameValue)
    def tuple_prior_tuples(self):
        """
        Returns
        -------
        tuple_prior_tuples: [(String, TuplePrior)]
        """
        return self.direct_tuples_with_type(TuplePrior)

    @property
    @cast_collection(PriorNameValue)
    def direct_prior_tuples(self):
        """
        Returns
        -------
        direct_priors: [(String, Prior)]
        """
        return self.direct_tuples_with_type(Prior)

    @property
    @cast_collection(DeferredNameValue)
    def direct_deferred_tuples(self):
        return self.direct_tuples_with_type(DeferredArgument)

    @property
    @cast_collection(PriorNameValue)
    def prior_tuples(self):
        """
        Returns
        -------
        priors: [(String, Prior))]
        """
        # noinspection PyUnresolvedReferences
        return self.attribute_tuples_with_type(Prior)

    @property
    @cast_collection(PriorModelNameValue)
    def direct_prior_model_tuples(self):
        return self.direct_tuples_with_type(AbstractPriorModel)

    def __eq__(self, other):
        return (
                isinstance(other, AbstractPriorModel)
                and self.direct_prior_model_tuples == other.direct_prior_model_tuples
        )

    @property
    @cast_collection(InstanceNameValue)
    def instance_tuples(self):
        """
        Returns
        -------
        instances: [(String, instance)]
        """
        return self.attribute_tuples_with_type(float, ignore_class=Prior)

    @property
    def prior_class_dict(self):
        from autofit.mapper.prior_model.annotation import AnnotationPriorModel

        d = {prior[1]: self.cls for prior in self.prior_tuples}
        for prior_model in self.prior_model_tuples:
            if not isinstance(prior_model[1], AnnotationPriorModel):
                d.update(prior_model[1].prior_class_dict)
        return d

    def _instance_for_arguments(self, arguments):
        raise NotImplementedError()

    def instance_for_arguments(
            self,
            arguments,
            assert_priors_in_limits=True
    ):
        """
        Create an instance of the model for a set of arguments
        Parameters
        ----------
        assert_priors_in_limits
            If true it is asserted that the physical values that replace piors are
            within their limits
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
        if assert_priors_in_limits:
            for prior, value in arguments.items():
                if isinstance(value, Number):
                    prior.assert_within_limits(value)
        return self._instance_for_arguments(
            arguments
        )

    @property
    def prior_count(self):
        return len(self.unique_prior_tuples)

    @property
    def promise_count(self):
        return len(self.unique_promise_tuples)

    @property
    def variable_promise_count(self):
        return len([
            value for key, value in
            self.unique_promise_tuples
            if not value.is_instance
        ])

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

    def __add__(self, other):
        result = copy.deepcopy(self)

        for key, value in other.__dict__.items():
            if not hasattr(result, key) or isinstance(value, Prior):
                setattr(result, key, value)
                continue
            result_value = getattr(result, key)
            if isinstance(value, AbstractPriorModel):
                if isinstance(result_value, AbstractPriorModel):
                    setattr(result, key, result_value + value)
                else:
                    setattr(result, key, value)

        return result

    def copy_with_fixed_priors(self, instance, excluded_classes=tuple()):
        """
        Recursively overwrite priors in the mapper with instance values from the
        instance except where the containing class is the descendant of a listed class.
        Parameters
        ----------
        excluded_classes
            Classes that should be left model
        instance
            The best fit from the previous phase
        """
        mapper = copy.deepcopy(self)
        transfer_classes(instance, mapper, excluded_classes)
        return mapper

    @property
    def path_priors_tuples(self):
        path_priors_tuples = self.path_instance_tuples_for_class(Prior)
        return sorted(path_priors_tuples, key=lambda item: item[1].id)

    def path_for_prior(self, prior: Prior) -> Optional[Tuple[str]]:
        """
        Find a path that points at the given tuple.
        Returns the first path or None if no path is found.
        Parameters
        ----------
        prior
            A prior representing what's known about some dimension of the model.
        Returns
        -------
        A path, a series of attributes that point to one location of the prior.
        """
        for path, path_prior in self.path_priors_tuples:
            if path_prior == prior:
                return path
        return None

    @property
    def path_float_tuples(self):
        return self.path_instance_tuples_for_class(float, ignore_class=Prior)

    @property
    def unique_prior_paths(self):
        unique = {item[1]: item for item in self.path_priors_tuples}.values()
        return [item[0] for item in sorted(unique, key=lambda item: item[1].id)]

    @property
    def prior_prior_model_dict(self):
        """
        Returns
        -------
        prior_prior_model_dict: {Prior: PriorModel}
            A dictionary mapping priors to associated prior models. Each prior will only
            have one prior model; if a prior is shared by two prior models then one of
            those prior models will be in this dictionary.
        """
        return {
            prior: prior_model[1]
            for prior_model in self.prior_model_tuples + [("model", self)]
            for _, prior in prior_model[1].prior_tuples
        }

    @property
    def info(self) -> str:
        """
        Use the priors that make up the model_mapper to generate information on each
        parameter of the overall model.
        This information is extracted from each priors *model_info* property.
        """
        from autofit.mapper.prior import AbstractPromise
        formatter = TextFormatter()

        for t in self.path_instance_tuples_for_class((
                Prior, float, AbstractPromise, tuple
        )):
            formatter.add(t)

        return formatter.text

    @property
    def parameter_names(self) -> [str]:
        """The param_names vector is a list each parameter's analysis_path, and is used
        for *GetDist* visualization.
        The parameter names are determined from the class instance names of the
        model_mapper. Latex tags are properties of each model class."""
        return [
            self.name_for_prior(
                prior
            )
            for _, prior
            in self.prior_tuples_ordered_by_id
        ]


def transfer_classes(instance, mapper, model_classes=None):
    """
    Recursively overwrite priors in the mapper with instance values from the
    instance except where the containing class is the descendant of a listed class.
    Parameters
    ----------
    model_classes
        Classes whose descendants should not be overwritten
    instance
        The best fit from the previous phase
    mapper
        The prior model from the previous phase
    """
    from autofit.mapper.prior_model.annotation import AnnotationPriorModel

    model_classes = model_classes or []
    for key, instance_value in instance.__dict__.items():
        try:
            mapper_value = getattr(mapper, key)
            if isinstance(mapper_value, Prior) or isinstance(
                    mapper_value, AnnotationPriorModel
            ):
                setattr(mapper, key, instance_value)
                continue
            if not any(isinstance(instance_value, cls) for cls in model_classes):
                try:
                    transfer_classes(instance_value, mapper_value, model_classes)
                except AttributeError:
                    setattr(mapper, key, instance_value)
        except AttributeError:
            pass
