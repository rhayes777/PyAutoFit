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

    def transformed_class(self):
        if self.__transformed_class is None:
            self.__transformed_class = self._transformed_class()
        return self.__transformed_class

    def _transformed_class(self):

        from .transformed import TransformedMessage

        projectionClass = (
            None if self.cls._projection_class is None
            else self.cls._projection_class.transformed(self.transform)
        )

        support = self.support or tuple(zip(*map(
            self.transform.inv_transform, map(np.array, zip(*self.cls._support))
        ))) if self.cls._support else self.cls._support

        if issubclass(self.cls, TransformedMessage):
            depth = self.cls._depth + 1
            clsname = self.clsname or f"Transformed{depth}{self.cls._Message.__name__}"

            # Don't doubly inherit if transforming already transformed message
            class Transformed(self.cls):  # type: ignore
                __qualname__ = clsname
                _depth = depth
        else:
            clsname = self.clsname or f"Transformed{self.cls.__name__}"

            class Transformed(TransformedMessage):  # type: ignore
                __qualname__ = clsname
                parameter_names = self.cls.parameter_names
                _depth = 1

        Transformed._Message = self.cls
        Transformed._transform = self.transform
        Transformed._support = support
        Transformed.__projection_class = projectionClass
        Transformed.__name__ = clsname

        return Transformed
