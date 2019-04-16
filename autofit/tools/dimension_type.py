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
    print(annotations)

    def map_to_type(name, value):
        print(name)
        print(value)
        print(annotations[name])
        try:
            return annotations[name](value)
        except KeyError:
            pass

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **{name: map_to_type(name, value) for name, value in kwargs.items()})

    return wrapper
