from typing import Union, Type, Optional, Tuple

import numpy as np

from autofit.messages.transform import AbstractDensityTransform


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
        return self.transformed_class()(
            *args, **kwargs
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
