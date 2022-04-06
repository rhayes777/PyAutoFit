from copy import copy
from typing import Union, Type, Optional, Tuple

import numpy as np

from autofit.mapper.prior.abstract import Prior
from autofit.messages.transform import AbstractDensityTransform, LinearShiftTransform
from .abstract import AbstractMessage


class TransformedWrapperInstance(Prior):
    """
    An instance of a transformed message. e.g. a UniformNormal message.

    This allows arbitrary transforms to be created on the fly while supporting
    the same interface. The true underlying message is still the same, but values
    computed from it are transformed to give the effect of a different distribution.
    """

    def value_for(self, unit: float) -> float:
        return self.instance().value_for(unit)

    def __init__(
            self,
            transformed_wrapper: "TransformedWrapper",
            *args,
            **kwargs
    ):
        """
        Parameters
        ----------
        transformed_wrapper
            Acts like a message class but provides a transformation on
            some underlying class.
        args
            Arguments required to instantiate the underlying message class
        kwargs
            Keyword arguments required to instantiate the underlying message
            class
        """
        super().__init__(
            id_=kwargs.get("id_")
        )
        self.transformed_wrapper = transformed_wrapper
        self.args = args
        self.kwargs = kwargs

        self._instance = None

    def project(
            self,
            samples: np.ndarray,
            log_weight_list: Optional[np.ndarray] = None,
            **kwargs,
    ):
        return self._new_for_base_message(
            self.transformed_wrapper.project(
                samples=samples,
                log_weight_list=log_weight_list,
                **kwargs,
            )
        )

    def _new_for_base_message(
            self,
            message
    ):
        """
        Create a new instance of this wrapper but change the parameters used
        to instantiate the underlying message. This is useful for retaining
        the same transform stack after recreating the underlying message during
        projection.
        """
        return type(self)(
            self.transformed_wrapper,
            *message.parameters,
            lower_limit=self.lower_limit,
            upper_limit=self.upper_limit,
            id_=self.instance().id
        )

    def __mul__(self, other):
        """
        Multiply this message by some other message. Effectively multiplies the
        underlying message whilst retaining the same transform stack.
        """
        if isinstance(
                other,
                TransformedWrapperInstance
        ):
            other = other.instance()
        return self._new_for_base_message(
            self.instance() * other
        )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        """
        Divide this message by some other message. Effectively divides the
        underlying message whilst retaining the same transform stack.
        """
        return self._new_for_base_message(
            self.instance() / other.instance()
        )

    def __sub__(self, other):
        instance = self.instance()
        if isinstance(
                other,
                TransformedWrapperInstance
        ):
            other = other.instance()
        return instance - other

    def __eq__(self, other):
        if not isinstance(
                other,
                TransformedWrapperInstance
        ):
            return False
        return other.instance() == self.instance()

    def __getattr__(self, item):
        """
        By default attributes are taken from the underlying message instance
        """
        return getattr(
            self.instance(),
            item
        )

    def copy(self):
        copied = copy(self)
        self._instance = None
        return copied

    @property
    def cls(self):
        return self.transformed_wrapper.cls

    def __hash__(self):
        return hash(self.instance())

    def __setstate__(self, state):
        self._instance = None
        self.__dict__.update(state)

    def __getstate__(self):
        """
        Representation of state of object for pickling excluding
        underlying instance as this can be reconstructed.
        """
        return {
            key: value
            for key, value
            in self.__dict__.items()
            if "_instance" != key
        }

    def instance(self):
        """
        An instance of the transformed message wrapped by this class.
        """
        if self._instance is None:
            cls = self.transformed_wrapper.transformed_class()
            self._instance = cls(
                *self.args,
                **self.kwargs,
            )
            self._instance.id = self.id
        return self._instance

    @property
    def factor(self):
        return self.instance().factor

    def from_mode(
            self,
            mode: np.ndarray,
            covariance: np.ndarray,
            **kwargs
    ):
        return self._new_for_base_message(
            self.transformed_wrapper.from_mode(
                mode,
                covariance,
                **kwargs
            )
        )


class TransformedWrapper:
    """
    A transformed message. This allows transformed messages to be created
    on the fly whilst retaining the same API.
    """

    InstanceWrapper = TransformedWrapperInstance

    def __init__(
            self,
            cls: Union[Type[AbstractMessage], "TransformedWrapper"],
            transform: AbstractDensityTransform,
            clsname: Optional[str] = None,
            support: Optional[Tuple[Tuple[float, float], ...]] = None,
    ):
        """
        Parameters
        ----------
        cls
            The class or TransformWrapper being transformed
        transform
            The transform applied
        clsname
            A custom name for the newly created class
        support
            The supported region
        """
        self.cls = cls
        self.transform = transform
        self.clsname = clsname
        self.support = support

        self.__transformed_class = None

    def __getattr__(self, item):
        """
        By default attributes values are taken from the underlying message class
        """
        return getattr(
            self.transformed_class(),
            item
        )

    def __call__(self, *args, **kwargs):
        """
        'Instantiate' the message. Creates an instance wrapper and
        passes arguments and keyword arguments to it for lazy instantiation
        of the underlying message.
        """
        return self.InstanceWrapper(
            self,
            *args,
            **kwargs
        )

    def transformed(
            self,
            transform: Union[
                AbstractDensityTransform,
                Type[AbstractDensityTransform]
            ],
            clsname: Optional[str] = None,
            support: Optional[Tuple[Tuple[float, float], ...]] = None,
            wrapper_cls=None
    ) -> "TransformedWrapper":
        """
        Apply a further transformation to an already transformed message
        """
        wrapper_cls = wrapper_cls or TransformedWrapper
        return wrapper_cls(
            cls=self,
            transform=transform,
            clsname=clsname,
            support=support,
        )

    def shifted(
            self,
            shift: float = 0,
            scale: float = 1,
            wrapper_cls=None,
    ):
        return self.transformed(
            LinearShiftTransform(shift=shift, scale=scale),
            clsname=f"Shifted{self.transformed_class().__name__}",
            wrapper_cls=wrapper_cls
        )

    def __getstate__(self):
        """
        The definition of the transformed class is not included in the state
        to circumvent an issue with pickling arbitrarily parameterised on-the-fly
        class definitions.

        The definition of the class is lazily created from the state of an instance
        as required.
        """
        return {
            key: value
            for key, value
            in self.__dict__.items()
            if "__transformed_class" not in key
        }

    def transformed_class(self):
        """
        The transformed class is lazily created
        """
        if self.__transformed_class is None:
            self.__transformed_class = self._transformed_class()
        return self.__transformed_class

    def _transformed_class(self):
        """
        The transformed class is lazily defined. This means it does not have to be
        pickled; it can entirely be derived from the state of the TransformedWrapper
        """
        from .transformed import TransformedMessage

        if isinstance(self.cls, TransformedWrapper):
            depth = self.cls._depth + 1
            clsname = self.clsname or f"Transformed{depth}{self.cls._Message.__name__}"

            # Don't doubly inherit if transforming already transformed message
            cls = self.cls.transformed_class()

            class Transformed(cls):
                __qualname__ = clsname
                _depth = depth
                _Message = cls
        else:
            cls = self.cls
            clsname = self.clsname or f"Transformed{self.cls.__name__}"

            class Transformed(TransformedMessage):  # type: ignore
                __qualname__ = clsname
                parameter_names = self.cls.parameter_names
                _depth = 1

        projectionClass = (
            None if cls._projection_class is None
            else cls._projection_class.transformed(self.transform)
        )

        support = self.support or tuple(zip(*map(
            self.transform.inv_transform, map(np.array, zip(*cls._support))
        ))) if cls._support else cls._support

        Transformed._transform = self.transform
        Transformed._support = support
        Transformed.__projection_class = projectionClass
        Transformed.__name__ = clsname
        Transformed._Message = cls

        return Transformed

    def __setstate__(self, state):
        self.__transformed_class = None
        self.__dict__.update(state)

    def from_mode(
            self,
            mode: np.ndarray,
            covariance: np.ndarray,
            **kwargs
    ):
        mode, jac = self._transform.transform_jac(mode)
        if covariance.shape != ():
            covariance = jac.quad(covariance)
        return self.cls.from_mode(mode, covariance, **kwargs)
