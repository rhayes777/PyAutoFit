import copy
import logging
from functools import wraps
from typing import Optional, Union, Tuple, List, Iterable, Type

from autofit.mapper.model_object import ModelObject
from autofit.mapper.prior_model.recursion import DynamicRecursionCache

logger = logging.getLogger(
    __name__
)


def frozen_cache(func):
    """
    Decorator that caches results from function calls when
    a model is frozen.

    Value is cached by function name, instance and arguments.

    Parameters
    ----------
    func
        Some function attached to a freezable, hashable object
        that takes hashable arguments

    Returns
    -------
    Function with cache
    """

    @wraps(func)
    def cache(self, *args, **kwargs):
        if hasattr(self, "_is_frozen") and self._is_frozen:
            key = (func.__name__, self, *args,) + tuple(
                kwargs.items()
            )
            if key not in self._frozen_cache:
                self._frozen_cache[
                    key
                ] = func(self, *args, **kwargs)
            return self._frozen_cache[
                key
            ]
        return func(self, *args, **kwargs)

    return cache


def assert_not_frozen(func):
    """
    Decorator that asserts a function is not called when an object
    is frozen. For example, it should not be possible to set an
    attribute on a frozen model as that might invalidate the results
    in the cache.

    Parameters
    ----------
    func
        Some function

    Raises
    ------
    AssertionError
        If the function is called when the object is frozen
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if "_is_frozen" not in filter(
                lambda arg: isinstance(arg, str),
                args
        ) and hasattr(self, "_is_frozen") and self._is_frozen:
            raise AssertionError(
                "Frozen models cannot be modified"
            )
        return func(self, *args, **kwargs)

    return wrapper


class AbstractModel(ModelObject):
    def __init__(self, label=None):
        super().__init__(label=label)
        self._is_frozen = False
        self._frozen_cache = dict()

    def freeze(self):
        """
        Freeze this object.

        A frozen object caches results for some function calls
        and does not allow its state to be modified.
        """
        logger.debug("Freezing model")
        tuples = self.direct_tuples_with_type(
            AbstractModel
        )
        for _, model in tuples:
            if model is not self:
                model.freeze()
        self._is_frozen = True

    def unfreeze(self):
        """
        Unfreeze this object. Allows modification and removes
        caches associated with some functions.
        """
        logger.debug("Thawing model")
        self._is_frozen = False
        tuples = self.direct_tuples_with_type(
            AbstractModel
        )
        for _, model in tuples:
            if model is not self:
                model.unfreeze()
        self._frozen_cache = dict()

    def __add__(self, other):
        instance = self.__class__()

        def add_items(item_dict):
            for key, value in item_dict.items():
                if isinstance(value, list) and hasattr(instance, key):
                    setattr(instance, key, getattr(instance, key) + value)
                else:
                    setattr(instance, key, value)

        add_items(self.__dict__)
        add_items(other.__dict__)
        return instance

    def copy(self):
        """
        Create a copy of the model. All priors remain equivalent - i.e. two
        copies of a model in a collection has the same prior count as a single
        model.
        """
        return copy.deepcopy(self)

    def object_for_path(
            self, path: Iterable[Union[str, int, type]]
    ) -> Union[object, List]:
        """
        Get the object at a given path.

        The path describes the location of some object in the model.

        String entries get an attribute.
        Int entries index an attribute.
        Type entries product a new ModelInstance which collates all of the instances
        of a given type in the path.

        Parameters
        ----------
        path
            A tuple describing the path to an object in the model tree

        Returns
        -------
        An object or Instance collating a collection of objects with a given type.
        """
        instance = self
        for name in path:
            if isinstance(name, int):
                instance = instance[name]
            elif isinstance(name, type):
                from autofit.mapper.prior_model.prior_model import PriorModel

                instances = [
                    instance
                    for _, instance in self.path_instance_tuples_for_class(name)
                ]
                instances += [
                    instance
                    for _, instance in self.path_instance_tuples_for_class(PriorModel)
                    if issubclass(instance.cls, name)
                ]
                instance = ModelInstance(instances)
            else:
                instance = getattr(instance, name)
        return instance

    @frozen_cache
    def path_instance_tuples_for_class(
            self,
            cls: Union[Tuple, Type],
            ignore_class: bool = None,
            ignore_children: bool = True
    ):
        """
        Tuples containing the path tuple and instance for every instance of the class
        in the model tree.

        Parameters
        ----------
        ignore_class
            Children of instances of this class are ignored
        ignore_children
            If true do not continue to recurse the children of an object once found
        cls
            The type to find instances of

        Returns
        -------
        path_instance_tuples: [((str,), object)]
            Tuples containing the path to and instance of objects of the given type.
        """
        return path_instances_of_class(
            self,
            cls,
            ignore_class=ignore_class,
            ignore_children=ignore_children
        )

    @frozen_cache
    def direct_tuples_with_type(self, class_type):
        return list(
            filter(
                lambda t: t[0] != "id" and not t[0].startswith("_") and isinstance(t[1], class_type),
                self.__dict__.items(),
            )
        )

    @frozen_cache
    def model_tuples_with_type(self, cls):
        """
        All models of the class in this model which have at least
        one free parameter, recursively.

        Parameters
        ----------
        cls
            The type of the model

        Returns
        -------
        Models with free parameters
        """
        from .prior_model.prior_model import PriorModel
        return [
            (path, model)
            for path, model
            in self.attribute_tuples_with_type(
                PriorModel
            )
            if issubclass(
                model.cls,
                cls
            ) and model.prior_count > 0
        ]

    @frozen_cache
    def attribute_tuples_with_type(
            self,
            class_type,
            ignore_class=None,
            ignore_children=True
    ) -> List[tuple]:
        """
        Tuples describing the name and instance for attributes in the model
        with a given type, recursively.

        Parameters
        ----------
        ignore_children
            If True then recursion stops at instances with the type
        class_type
            The type of the objects to find
        ignore_class
            Any classes which should not be recursively searched

        Returns
        -------
        Tuples containing the name and instance of each attribute with the type
        """
        return [
            (path[-1] if len(path) > 0 else "", value)
            for path, value in self.path_instance_tuples_for_class(
                class_type,
                ignore_class=ignore_class,
                ignore_children=ignore_children
            )
        ]


@DynamicRecursionCache()
def path_instances_of_class(
        obj,
        cls: type,
        ignore_class: Optional[Union[type, Tuple[type]]] = None,
        ignore_children: bool = False
):
    """
    Recursively search the object for instances of a given class

    Parameters
    ----------
    obj
        The object to recursively search
    cls
        The type to search for
    ignore_class
        A type or tuple of classes to skip
    ignore_children
        If true stop recursion at found objects

    Returns
    -------
    instance of type
    """
    if ignore_class is not None and isinstance(obj, ignore_class):
        return []

    results = []
    if isinstance(obj, cls):
        results.append((tuple(), obj))
        if ignore_children:
            return results
    try:
        from autofit.mapper.prior_model.annotation import AnnotationPriorModel

        if isinstance(obj, dict):
            d = obj
        else:
            d = obj.__dict__

        for key, value in d.items():
            if key.startswith("_"):
                continue
            for item in path_instances_of_class(
                    value,
                    cls,
                    ignore_class=ignore_class,
                    ignore_children=ignore_children
            ):
                if isinstance(value, AnnotationPriorModel):
                    path = (key,)
                else:
                    path = (key, *item[0])
                results.append((path, item[1]))
        return results
    except (AttributeError, TypeError):
        return results


class ModelInstance(AbstractModel):
    """
    An object to hold model instances produced by providing arguments to a model mapper.

    @DynamicAttrs
    """

    def __init__(self, items=None):
        super().__init__()
        if isinstance(items, list):
            for i, item in enumerate(items):
                self[i] = item
        if isinstance(items, dict):
            for key, value in items.items():
                self[key] = value

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __getitem__(self, item):
        if isinstance(item, int):
            return list(self.values())[item]
        if isinstance(item, slice):
            return ModelInstance(
                list(self.values())[item]
            )
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def items(self):
        return self.dict.items()

    @property
    def dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in (
                "id",
                "component_number",
                "item_number"
            ) and not (
                    isinstance(key, str)
                    and key.startswith("_")
            )
        }

    def values(self):
        return self.dict.values()

    def __len__(self):
        return len(self.values())

    def as_model(self, model_classes=tuple()):
        from autofit.mapper.prior_model.abstract import AbstractPriorModel

        return AbstractPriorModel.from_instance(self, model_classes)
