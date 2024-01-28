from autoconf.tools.decorators import CachedProperty


class DerivedQuantityDecorator(CachedProperty):
    """
    A derived quantity is an important quantity that is computed from a model instance.

    This decorator makes a method into a property and indicates that autofit should
    track the quantity.

    Once a quantity is computed, it is cached and not recomputed.
    """

    def __set__(self, obj, value):
        obj.__dict__[self.func.__name__] = value


derived_quantity = DerivedQuantityDecorator
