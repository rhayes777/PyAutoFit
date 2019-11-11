from functools import wraps


class RecursionPromise:
    pass


cache = {}


def dynamic_recursion_cache(
        func
):
    @wraps(func)
    def wrapper(item):
        item_id = id(item)
        if item_id in cache:
            return cache[item_id]
        cache[
            item_id
        ] = RecursionPromise()
        return func(item)

    return wrapper


class Wrapper:
    def __init__(
            self,
            item
    ):
        self.item = item


class A:
    def __init__(
            self,
            b=None
    ):
        self.b = b


class B:
    def __init__(
            self,
            a=None
    ):
        self.a = a


@dynamic_recursion_cache
def dict_recurse(
        item
):
    try:
        for key, value in item.__dict__.items():
            setattr(
                item,
                key,
                dict_recurse(value)
            )
    except AttributeError:
        pass
    return Wrapper(item)


def test_basic():
    a = A(B())
    a.b.a = a
    result = dict_recurse(
        a
    )
    assert isinstance(result.item.b.item.a, Wrapper)
