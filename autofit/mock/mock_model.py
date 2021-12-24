class MockListClass:
    def __init__(self, ls: list):
        self.ls = ls


class MockClassx2:
    def __init__(self, one=1, two=2):
        self.one = one
        self.two = two


class MockClassx2NoSuperScript:
    def __init__(self, one=1, two=2):
        self.one = one
        self.two = two


class MockClassx4:
    def __init__(self, one=1, two=2, three=3, four=4):
        self.one = one
        self.two = two
        self.three = three
        self.four = four


class MockClassx3(MockClassx4):
    def __init__(self, one=1, two=2, three=3):
        super().__init__(one, two, three)


class MockClassx2Tuple:
    def __init__(self, one_tuple=(0.0, 0.0)):
        """Abstract GeometryProfile, describing an object with y, x cartesian
        coordinates """
        self.one_tuple = one_tuple

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class MockClassx3TupleFloat:
    def __init__(self, one_tuple=(0.0, 0.0), two=0.1):
        self.one_tuple = one_tuple
        self.two = two


class MockClassRelativeWidth:
    def __init__(self, one, two, three):
        self.one = one
        self.two = two
        self.three = three


class MockClassInf:
    def __init__(self, one, two):
        self.one = one
        self.two = two


class MockComplexClass:
    def __init__(self, simple: MockClassx2):
        self.simple = simple


class MockDeferredClass:
    def __init__(self, one, two):
        self.one = one
        self.two = two


class MockWithFloat:
    def __init__(self, value):
        self.value = value


class MockWithTuple:
    def __init__(self, tup=(0.0, 0.0)):
        self.tup = tup


class MockComponents:
    def __init__(
            self,
            components_0: list = None,
            components_1: list = None,
            parameter=None,
            **kwargs
    ):
        self.parameter = parameter
        self.group_0 = components_0
        self.group_1 = components_1
        self.kwargs = kwargs
