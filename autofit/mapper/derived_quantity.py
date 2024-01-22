from autoconf.tools.decorators import CachedProperty


class DerivedQuantityDecorator(CachedProperty):
    def __set__(self, obj, value):
        obj.__dict__[self.func.__name__] = value


derived_quantity = DerivedQuantityDecorator
