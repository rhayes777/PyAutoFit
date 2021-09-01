from typing import Union, Type, Optional, Tuple

import numpy as np

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
        if "__transformed_class" in item:
            raise AttributeError()
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


class TransformedWrapperInstance:
    def __init__(
            self,
            transformed_wrapper,
            *args,
            **kwargs
    ):
        self.transformed_wrapper = transformed_wrapper
        self.args = args
        self.kwargs = kwargs

        self._instance = None

    def __eq__(self, other):
        return other.instance() == self.instance()

    def __getattr__(self, item):
        print(item)
        if item == "_instance":
            return None
        if item == "transformed_wrapper":
            raise AttributeError()
        return getattr(
            self.instance(),
            item
        )

    def __setstate__(self, state):
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
                **self.kwargs
            )
        return self._instance
