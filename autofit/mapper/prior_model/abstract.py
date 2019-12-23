import copy
import inspect

import numpy as np

import autofit.mapper.model
import autofit.mapper.model_mapper
import autofit.mapper.prior_model.collection
from autofit import conf
from autofit import exc
from autofit.mapper.model import AbstractModel
from autofit.mapper.prior_model import dimension_type as dim
from autofit.mapper.prior_model.deferred import DeferredArgument
from autofit.mapper.prior_model.prior import GaussianPrior
from autofit.mapper.prior_model.prior import (
    cast_collection,
    PriorNameValue,
    TuplePrior,
    Prior,
    DeferredNameValue,
)
from autofit.mapper.prior_model.prior import instanceNameValue
from autofit.mapper.prior_model.recursion import DynamicRecursionCache
from autofit.mapper.prior_model.util import PriorModelNameValue
from autofit.tools.text_formatter import TextFormatter


class AbstractPriorModel(AbstractModel):
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
            obj = object.__new__(
                autofit.mapper.prior_model.collection.CollectionPriorModel
            )
            obj.__init__(t)
        else:
            obj = t
        return obj

    def instance_from_unit_vector(self, unit_vector):
        """
        Creates a ModelInstance, which has an attribute and class instance corresponding
        to every PriorModel attributed to this instance.

        This method takes as input a unit vector of parameter values, converting each to
        physical values via their priors.

        Parameters
        ----------
        unit_vector: [float]
            A vector of physical parameter values.

        Returns
        -------
        model_instance : autofit.mapper.model.ModelInstance
            An object containing reconstructed model_mapper instances

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

        return self.instance_for_arguments(arguments)

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
        from autofit import Promise

        return {
            prior_tuple[1]: prior_tuple
            for prior_tuple in self.attribute_tuples_with_type(Promise)
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

    def physical_vector_from_hypercube_vector(self, hypercube_vector):
        """
        Parameters
        ----------
        hypercube_vector: [float]
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
                hypercube_vector,
            )
        )

    @property
    def random_physical_vector_from_priors(self):
        """
        Returns
        -------
        physical_values: [float]
            A list of physical values constructed by taking the mean possible value from
            each prior.
        """

        while True:

            physical_vector = self.physical_vector_from_hypercube_vector(
                list(np.random.random(self.prior_count))
            )

            try:
                self.instance_from_physical_vector(physical_vector=physical_vector)
                return physical_vector
            except exc.PriorLimitException:
                pass

    @property
    def physical_values_from_prior_medians(self):
        """
        Returns
        -------
        physical_values: [float]
            A list of physical values constructed by taking the mean possible value from
            each prior.
        """
        return self.physical_vector_from_hypercube_vector(
            [0.5] * len(self.unique_prior_tuples)
        )

    def instance_from_physical_vector(self, physical_vector):
        """
        Creates a ModelInstance, which has an attribute and class instance corresponding
        to every PriorModel attributed to this instance.

        This method takes as input a physical vector of parameter values, thus omitting
        the use of priors.

        Parameters
        ----------
        physical_vector: [float]
            A unit hypercube vector

        Returns
        -------
        model_instance : autofit.mapper.model.ModelInstance
            An object containing reconstructed model_mapper instances

        """
        arguments = dict(
            map(
                lambda prior_tuple, physical_unit: (prior_tuple.prior, physical_unit),
                self.prior_tuples_ordered_by_id,
                physical_vector,
            )
        )

        return self.instance_for_arguments(arguments)

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

    def mapper_from_gaussian_tuples(self, tuples, a=None, r=None):
        """
        Creates a new model mapper from a list of floats describing the mean values
        of gaussian priors. The widths of the new priors are taken from the
        width_config. The new gaussian priors must be provided in the same order as
        the priors associated with model.

        If a is not None then all priors are created with an absolute width of a.

        If r is not None then all priors are created with a relative width of r.

        Parameters
        ----------
        r
            The relative width to be assigned to gaussian priors
        a
            The absolute width to be assigned to gaussian priors
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
            mean = tuples[i][0]
            if a is not None and r is not None:
                raise exc.PriorException(
                    "Width of new priors cannot be both relative and absolute."
                )
            if a is not None:
                width_type = "a"
                value = a
            elif r is not None:
                width_type = "r"
                value = r
            else:
                width_type, value = conf.instance.prior_width.get_for_nearest_ancestor(
                    cls, prior_tuple.name
                )
            if width_type == "r":
                width = value * mean
            elif width_type == "a":
                width = value
            else:
                raise exc.PriorException(
                    "Prior widths must be relative 'r' or absolute 'a' e.g. a, 1.0"
                )
            if isinstance(prior, GaussianPrior):
                limits = (prior.lower_limit, prior.upper_limit)
            else:
                limits = conf.instance.prior_limit.get_for_nearest_ancestor(
                    cls, prior_tuple.name
                )
            arguments[prior] = GaussianPrior(mean, max(tuples[i][1], width), *limits)

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
            result = autofit.mapper.prior_model.collection.CollectionPriorModel(
                [
                    AbstractPriorModel.from_instance(item, model_classes=model_classes)
                    for item in instance
                ]
            )
        elif isinstance(instance, autofit.mapper.model.ModelInstance):
            result = autofit.mapper.model_mapper.ModelMapper()
            for key, value in instance.dict.items():
                setattr(
                    result,
                    key,
                    AbstractPriorModel.from_instance(
                        value, model_classes=model_classes
                    ),
                )
        elif isinstance(instance, dict):
            result = autofit.mapper.prior_model.collection.CollectionPriorModel(
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
    @cast_collection(instanceNameValue)
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
    @cast_collection(instanceNameValue)
    def instance_tuples(self):
        """
        Returns
        -------
        instances: [(String, instance)]
        """
        return self.attribute_tuples_with_type(float, ignore_class=Prior)

    @property
    def prior_class_dict(self):
        raise NotImplementedError()

    def instance_for_arguments(self, arguments):
        raise NotImplementedError()

    @property
    def prior_count(self):
        return len(self.unique_prior_tuples)

    @property
    def promise_count(self):
        return len(self.unique_promise_tuples)

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
    def info(self):
        """
        Use the priors that make up the model_mapper to generate information on each
        parameter of the overall model.

        This information is extracted from each priors *model_info* property.
        """
        formatter = TextFormatter()

        for t in self.path_priors_tuples + self.path_float_tuples:
            formatter.add(t)

        return formatter.text

    @property
    def param_names(self):
        """The param_names vector is a list each parameter's analysis_path, and is used
        for *GetDist* visualization.

        The parameter names are determined from the class instance names of the
        model_mapper. Latex tags are properties of each model class."""

        return [
            self.name_for_prior(prior)
            for prior in sorted(self.priors, key=lambda prior: prior.id)
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
