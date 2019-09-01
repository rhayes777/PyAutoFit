import copy
import inspect

import autofit.mapper.model
import autofit.mapper.model_mapper
import autofit.mapper.prior_model.collection
from autofit.mapper.model import AbstractModel
from autofit.mapper.prior_model.dimension_type import DimensionType
from autofit.mapper.prior_model.prior import Prior
from autofit.mapper.prior_model.prior import cast_collection, PriorNameValue, ConstantNameValue
from autofit.mapper.prior_model.util import PriorModelNameValue


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
            obj = object.__new__(autofit.mapper.prior_model.collection.CollectionPriorModel)
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
                    prior_tuple.prior.value_for(unit)
                ),
                self.prior_tuples_ordered_by_id,
                unit_vector
            )
        )

        return self.instance_for_arguments(arguments)

    @property
    @cast_collection(PriorNameValue)
    def prior_tuples_ordered_by_id(self):
        """
        Returns
        -------
        priors: [Prior]
            An ordered list of unique priors associated with this mapper
        """
        return sorted(list(self.prior_tuples),
                      key=lambda prior_tuple: prior_tuple.prior.id)

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
    def from_instance(
            instance,
            variable_classes=tuple()
    ):
        """
        Recursively create an prior object model from an object model.

        Parameters
        ----------
        variable_classes
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
                    AbstractPriorModel.from_instance(
                        item,
                        variable_classes=variable_classes
                    )
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
                        value,
                        variable_classes=variable_classes
                    )
                )
        elif isinstance(instance, dict):
            result = autofit.mapper.prior_model.collection.CollectionPriorModel(
                {
                    key: AbstractPriorModel.from_instance(
                        value,
                        variable_classes=variable_classes
                    )
                    for key, value
                    in instance.items()
                }
            )
        elif isinstance(instance, DimensionType):
            return instance
        else:
            from .prior_model import PriorModel
            try:
                result = PriorModel(
                    instance.__class__,
                    **{
                        key: AbstractPriorModel.from_instance(
                            value,
                            variable_classes=variable_classes
                        )
                        for key, value
                        in instance.__dict__.items()
                        if key != "cls"
                    }
                )

            except AttributeError:
                return instance
        if any([
            isinstance(instance, cls)
            for cls in variable_classes
        ]):
            return result.as_variable()
        return result

    @property
    @cast_collection(PriorNameValue)
    def direct_prior_tuples(self):
        return self.tuples_with_type(Prior)

    @property
    @cast_collection(ConstantNameValue)
    def direct_constant_tuples(self):
        return self.tuples_with_type(float)

    @property
    def flat_prior_model_tuples(self):
        """
        Returns
        -------
        prior_models: [(str, AbstractPriorModel)]
            A list of prior models associated with this instance
        """
        raise NotImplementedError(
            "PriorModels must implement the flat_prior_models property")

    @property
    @cast_collection(PriorModelNameValue)
    def prior_model_tuples(self):
        return self.tuples_with_type(AbstractPriorModel)

    @property
    def prior_models(self):
        return [item[1] for item in self.prior_model_tuples]

    @property
    def prior_tuples(self):
        raise NotImplementedError()

    @property
    @cast_collection(PriorModelNameValue)
    def direct_prior_model_tuples(self):
        return [(name, value) for name, value in self.__dict__.items() if
                isinstance(value, AbstractPriorModel)]

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

    def copy_with_fixed_priors(
            self,
            instance,
            excluded_classes=tuple()
    ):
        """
        Recursively overwrite priors in the mapper with constant values from the
        instance except where the containing class is the descendant of a listed class.

        Parameters
        ----------
        excluded_classes
            Classes that should be left variable
        instance
            The best fit from the previous phase
        """
        mapper = copy.deepcopy(self)
        transfer_classes(instance, mapper, excluded_classes)
        return mapper


def transfer_classes(instance, mapper, variable_classes=None):
    """
    Recursively overwrite priors in the mapper with constant values from the
    instance except where the containing class is the descendant of a listed class.

    Parameters
    ----------
    variable_classes
        Classes whose descendants should not be overwritten
    instance
        The best fit from the previous phase
    mapper
        The prior variable from the previous phase
    """
    from autofit.mapper.prior_model.annotation import AnnotationPriorModel
    variable_classes = variable_classes or []
    for key, instance_value in instance.__dict__.items():
        try:
            mapper_value = getattr(mapper, key)
            if isinstance(
                    mapper_value,
                    Prior
            ) or isinstance(
                mapper_value,
                AnnotationPriorModel
            ):
                setattr(mapper, key, instance_value)
                continue
            if not any(
                    isinstance(
                        instance_value,
                        cls
                    )
                    for cls in variable_classes
            ):
                try:
                    transfer_classes(
                        instance_value,
                        mapper_value,
                        variable_classes
                    )
                except AttributeError:
                    setattr(mapper, key, instance_value)
        except AttributeError:
            pass
