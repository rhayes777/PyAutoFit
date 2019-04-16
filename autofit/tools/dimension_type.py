import inspect
import typing

from decorator import decorator


class DimensionType(float):
    def __new__(cls, value):
        # noinspection PyArgumentList
        return float.__new__(cls, value)

    def __init__(self, value):
        float.__init__(value)


@decorator
def map_types(func, self, *args, **kwargs):
    annotations = inspect.getfullargspec(func).annotations

    def map_to_type(value, name=None, position=None):
        arg_type = None
        if name is not None:
            try:
                arg_type = annotations[name]
            except KeyError:
                pass
        if position is not None:
            arg_type = list(annotations.values())[position]
        if isinstance(arg_type, typing.TupleMeta):
            return tuple(element_type(element_value) for element_type, element_value in zip(arg_type.__args__, value))

        return arg_type(value)

    return func(self, *[map_to_type(value, position=index) for index, value in enumerate(args)],
                **{name: map_to_type(value, name=name) for name, value in kwargs.items()})
