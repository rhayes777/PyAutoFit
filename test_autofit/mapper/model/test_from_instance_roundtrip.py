"""
Regression tests for issue #1607.

``AbstractPriorModel.from_instance`` used to copy every entry of an instance's
``__dict__`` onto the model, including attributes an ``__init__`` derives from
its arguments. Those attributes were then passed back to the constructor by
``ModelObject.from_dict``, raising ``TypeError`` and breaking aggregator loading
of any ``model.json`` written from such a model.

The classes below are defined at module level so that the ``class_path`` written
into the dictionary resolves when it is read back.
"""

import autofit as af


class Shear:
    """
    A class with a derived attribute (``centre``) and a property, neither of
    which is a constructor argument.
    """

    def __init__(self, gamma_1=0.0, gamma_2=0.0):
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        # Derived: recomputed by __init__, never passed to it.
        self.centre = (0.0, 0.0)

    @property
    def magnitude(self):
        return (self.gamma_1**2 + self.gamma_2**2) ** 0.5


class Isothermal:
    """
    A tuple-valued constructor argument alongside a derived scalar.
    """

    def __init__(self, centre=(0.0, 0.0), einstein_radius=1.0):
        self.centre = centre
        self.einstein_radius = einstein_radius
        self.slope = 2.0


class WithKwargs:
    """
    A constructor taking ``**kwargs`` and storing them as attributes. Every
    attribute may legitimately be passed back, so none are filtered.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Outer:
    """
    An outer class holding a derived-attribute instance as an argument.
    """

    def __init__(self, shear=None, scale=1.0):
        self.shear = shear or Shear()
        self.scale = scale
        self.total = scale * 2.0


def test_derived_attributes_are_not_model_arguments():
    model = af.Model.from_instance(Shear(gamma_1=0.02, gamma_2=0.03))

    assert {name for name, _ in model.items()} == {"gamma_1", "gamma_2"}
    assert set(model.dict()["arguments"]) == {"gamma_1", "gamma_2"}
    assert not hasattr(model, "centre")
    assert not hasattr(model, "magnitude")


def test_derived_attribute_round_trip():
    instance = af.Model.from_dict(
        af.Model.from_instance(Shear(gamma_1=0.02, gamma_2=0.03)).dict()
    )

    assert isinstance(instance, Shear)
    assert instance.gamma_1 == 0.02
    assert instance.gamma_2 == 0.03
    # Recomputed by __init__ rather than carried through the dictionary.
    assert instance.centre == (0.0, 0.0)
    assert instance.magnitude == (0.02**2 + 0.03**2) ** 0.5


def test_tuple_argument_round_trip():
    model = af.Model.from_instance(
        Isothermal(centre=(1.0, 2.0), einstein_radius=1.5)
    )

    assert set(model.dict()["arguments"]) == {"centre", "einstein_radius"}

    instance = af.Model.from_dict(model.dict())

    assert isinstance(instance, Isothermal)
    assert instance.centre == (1.0, 2.0)
    assert instance.einstein_radius == 1.5
    assert instance.slope == 2.0


def test_kwargs_constructor_keeps_every_attribute():
    model = af.Model.from_instance(WithKwargs(one=1.0, two=2.0))

    assert set(model.dict()["arguments"]) == {"one", "two"}

    instance = af.Model.from_dict(model.dict())

    assert isinstance(instance, WithKwargs)
    assert instance.one == 1.0
    assert instance.two == 2.0


def test_nested_derived_attribute_round_trip():
    model = af.Model.from_instance(
        Outer(shear=Shear(gamma_1=0.04, gamma_2=0.05), scale=3.0)
    )

    assert set(model.dict()["arguments"]) == {"shear", "scale"}

    instance = af.Model.from_dict(model.dict())

    assert isinstance(instance, Outer)
    assert isinstance(instance.shear, Shear)
    assert instance.shear.gamma_1 == 0.04
    assert instance.shear.gamma_2 == 0.05
    assert instance.shear.centre == (0.0, 0.0)
    assert instance.scale == 3.0
    assert instance.total == 6.0
