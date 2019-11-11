from functools import wraps


class RecursionPromise:
    pass


cache = {}
result_cache = {}


def replace_promise(
        promise,
        obj,
        true_value
):
    if isinstance(
            obj,
            list
    ):
        return [
            replace_promise(
                promise,
                item,
                true_value
            )
            for item in obj
        ]
    if isinstance(
            obj,
            dict
    ):
        return {
            key: replace_promise(
                promise,
                value,
                true_value
            )
            for key, value
            in obj.items()
        }
    if obj is promise:
        return true_value
    try:
        for key, value in obj.__dict__.items():
            setattr(obj, key, replace_promise(
                promise,
                value,
                true_value
            ))
    except AttributeError:
        pass
    return obj


def dynamic_recursion_cache(
        func
):
    @wraps(func)
    def wrapper(item):
        item_id = id(item)
        if item_id in cache:
            return cache[item_id]
        recursion_promise = RecursionPromise()
        cache[
            item_id
        ] = recursion_promise
        result = func(item)
        result = replace_promise(
            recursion_promise,
            result,
            result
        )
        del cache[item_id]
        return result

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
