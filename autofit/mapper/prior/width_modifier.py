import inspect
import sys

from autoconf import conf


class WidthModifier:
    def __init__(self, value):
        self.value = float(value)

    @classmethod
    def name_of_class(cls) -> str:
        """
        A string name for the class, with the prior suffix removed.
        """
        return cls.__name__.replace("WidthModifier", "")

    @classmethod
    def from_dict(cls, width_modifier_dict):
        return width_modifier_type_dict[width_modifier_dict["type"]](
            value=width_modifier_dict["value"]
        )

    @property
    def dict(self):
        return {"type": self.name_of_class(), "value": self.value}

    @staticmethod
    def for_class_and_attribute_name(cls, attribute_name):
        prior_dict = conf.instance.prior_config.for_class_and_suffix_path(
            cls, [attribute_name, "width_modifier"]
        )
        return WidthModifier.from_dict(prior_dict)

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.value == other.value


class RelativeWidthModifier(WidthModifier):
    def __call__(self, mean):
        return self.value * mean


class AbsoluteWidthModifier(WidthModifier):
    def __call__(self, _):
        return self.value


width_modifier_type_dict = {
    obj.name_of_class(): obj
    for _, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isclass(obj)
}
