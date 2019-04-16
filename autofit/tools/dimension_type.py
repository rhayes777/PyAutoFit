import functools
import inspect


class DimensionType(float):
    def __new__(cls, value):
        # noinspection PyArgumentList
        return float.__new__(cls, value)

    def __init__(self, value):
        float.__init__(value)


def map_types(func):
    annotations = inspect.getfullargspec(func).annotations

    def map_to_type(value, name=None, position=None):
        if name is not None:
            try:
                return annotations[name](value)
            except KeyError:
                pass
        if position is not None:
            return list(annotations.values())[position](value)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *[map_to_type(value, position=index) for index, value in enumerate(args)],
                    **{name: map_to_type(value, name=name) for name, value in kwargs.items()})

    return wrapper
