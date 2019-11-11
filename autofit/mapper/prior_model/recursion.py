from functools import wraps


class RecursionPromise:
    pass


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
    except (AttributeError, TypeError):
        pass
    return obj


class DynamicRecursionCache:
    def __init__(self):
        self.cache = dict()

    def __call__(
            self,
            func
    ):
        @wraps(func)
        def wrapper(
                *args,
                **kwargs
        ):
            item_id = sum(
                id(item)
                for item in (
                        list(args) +
                        list(kwargs.items())
                )
            )
            if item_id in self.cache:
                return self.cache[item_id]
            recursion_promise = RecursionPromise()
            self.cache[
                item_id
            ] = recursion_promise
            result = func(
                *args,
                **kwargs
            )
            result = replace_promise(
                recursion_promise,
                result,
                result
            )
            del self.cache[item_id]
            return result

        return wrapper
