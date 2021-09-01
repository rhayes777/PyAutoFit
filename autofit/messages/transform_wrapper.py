from typing import Union, Type, Optional, Tuple

import numpy as np

from autofit.mapper.prior.abstract import Prior
from autofit.messages.transform import AbstractDensityTransform, LinearShiftTransform


class TransformedWrapper:
    def __init__(
            self,
            cls,
            transform: Union[
                AbstractDensityTransform,
                Type[AbstractDensityTransform]
            ],
            clsname: Optional[str] = None,
            support: Optional[Tuple[Tuple[float, float], ...]] = None,
    ):
        self.cls = cls
        self.transform = transform
        self.clsname = clsname
        self.support = support

        self.__transformed_class = None

    def __getattr__(self, item):
        return getattr(
            self.transformed_class(),
            item
        )

    def __call__(self, *args, **kwargs):
        return TransformedWrapperInstance(
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
    ):
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
        return {
            key: value
            for key, value
            in self.__dict__.items()
            if "__transformed_class" not in key
        }

    def transformed_class(self):
        if self.__transformed_class is None:
            self.__transformed_class = self._transformed_class()
        return self.__transformed_class

    def _transformed_class(self):

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


class TransformedWrapperInstance(Prior):
    def value_for(self, unit: float) -> float:
        return self.instance().value_for(unit)

    def __init__(
            self,
            transformed_wrapper,
            *args,
            **kwargs
    ):
        super().__init__(
            id_=kwargs.get("id_")
        )
        self.transformed_wrapper = transformed_wrapper
        self.args = args
        self.kwargs = kwargs

        self._instance = None

    def _new_for_base_message(
            self,
            message
    ):
        return type(self)(
            self.transformed_wrapper,
            *message.parameters,
            id_=self.instance().id
        )

    def __mul__(self, other):
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
        return other.instance() == self.instance()

    def __getattr__(self, item):
        return getattr(
            self.instance(),
            item
        )

    def __hash__(self):
        return hash(self.instance())

    def __setstate__(self, state):
        self._instance = None
        self.__dict__.update(state)

    def __getstate__(self):
        return {
            key: value
            for key, value
            in self.__dict__.items()
            if "_instance" != key
        }

    def instance(self):
        if self._instance is None:
            cls = self.transformed_wrapper.transformed_class()
            self._instance = cls(
                *self.args,
                **self.kwargs,
            )
            self._instance.id = self.id
        return self._instance
