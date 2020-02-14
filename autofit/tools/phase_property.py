class PhaseProperty:
    def __init__(self, name):
        """
        A phase property is a named property of a phase in a pipeline. It implemented setters and getters that
        allow it to associated values with the instance or model object depending on the type of those
        values. Note that this functionality may be better handled by the model mapper.

        Parameters
        ----------
        name: str
            The name of this property

        Examples
        --------
        class Phase:
            my_property = PhaseProperty("my_property")
            def __init__(self, my_property):
                self.my_property = my_property
        """
        self.name = name

    def fget(self, obj):
        return getattr(obj.model, self.name)

    def fset(self, obj, value):
        setattr(obj.model, self.name, value)

    def fdel(self, obj):
        delattr(obj.model, self.name)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.fget(obj)

    def __set__(self, obj, value):
        self.fset(obj, value)

    def __delete__(self, obj):
        return self.fdel(obj)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)
