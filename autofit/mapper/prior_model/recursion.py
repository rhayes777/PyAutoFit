from functools import wraps


class RecursionPromise:
    """
    A placeholder standing in for an object that is still being computed.

    ``used`` records whether this placeholder was ever handed back to a caller.
    The only way to obtain one is a cache hit in
    :meth:`DynamicRecursionCache.__call__`'s wrapper, which sets the flag, so a
    promise with ``used is False`` was never seen by any code outside the
    wrapper that created it. No object in the result can then hold a reference
    to it, and the ``replace_promise`` traversal below is provably a no-op --
    one that still walks (and ``setattr``s its way through) every object
    reachable from the result, over a graph that grows with the model.

    ``__slots__`` is load-bearing twice over: it keeps the flag off the
    instance ``__dict__``, and it means ``replace_promise`` takes its
    ``AttributeError`` branch immediately on a promise that *is* live rather
    than iterating a dict of its own.
    """

    __slots__ = ("used",)

    def __init__(self):
        self.used = False


def replace_promise(promise: RecursionPromise, obj, true_value, seen_objects=None):
    """
    Traverse the object replacing any identity of the promise with the true value

    Parameters
    ----------
    promise
        A placeholder for an object that had not been computed at the time some part of the object was computed
    obj
        An object computed that may contain Promises
    true_value
        The true value associated with the promise
    seen_objects
        A set of ids of objects that have already been checked in this promise replacement

    Returns
    -------
    obj
        The object with any identities of the Promise replaced
    """
    seen_objects = seen_objects or set()
    if id(obj) in seen_objects:
        return obj

    seen_objects.add(id(obj))

    if isinstance(obj, list):
        return [
            replace_promise(promise, item, true_value, seen_objects=seen_objects)
            for item in obj
        ]
    if isinstance(obj, dict):
        return {
            key: replace_promise(promise, value, true_value, seen_objects=seen_objects)
            for key, value in obj.items()
        }

    if obj is promise:
        return true_value
    try:
        for key, value in list(obj.__dict__.items()):
            setattr(
                obj,
                key,
                replace_promise(promise, value, true_value, seen_objects=seen_objects),
            )
    except (AttributeError, TypeError):
        pass
    return obj


class DynamicRecursionCache:
    def __init__(self):
        """
        A decorating class that prevents infinite loops when recursing graphs by attaching placeholders
        """
        self.cache = dict()

    def __call__(self, func):
        """
        Decorate the function to prevent recursion.

        When the function is called with a set of arguments, A, a Promise is stored for that set of arguments in the
        cache. If the function is called again with that set of arguments then the Promise is returned. When the
        function itself returns a value any identity of the Promise is replaced by the actual value returned.
        """

        @wraps(func)
        def wrapper(item, *args, **kwargs):
            item_id = id(item)

            if item_id in self.cache:
                recursion_promise = self.cache[item_id]
                # The promise escapes here and only here, so this is the one
                # place that can put it inside the result being built.
                recursion_promise.used = True
                return recursion_promise

            recursion_promise = RecursionPromise()
            self.cache[item_id] = recursion_promise
            result = func(item, *args, **kwargs)

            if recursion_promise.used:
                result = replace_promise(recursion_promise, result, result)

            del self.cache[item_id]
            return result

        return wrapper
