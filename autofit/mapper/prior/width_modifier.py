import inspect
import logging
import sys

from autoconf import conf
from autoconf.exc import ConfigException

logger = logging.getLogger(
    __name__
)


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
    def for_class_and_attribute_name(
            cls: type,
            attribute_name: str
    ) -> "WidthModifier":
        """
        Search prior configuration for a WidthModifier.

        If no configuration is found a RelativeWidthModifier
        with a value of 0.5 is returned.

        Parameters
        ----------
        cls
            The class to which an attribute belongs
        attribute_name
            The name of an attribute

        Returns
        -------
        An object that ensures the prior derived from the
        posterior of a search is sufficiently wide.
        """
        try:
            prior_dict = conf.instance.prior_config.for_class_and_suffix_path(
                cls, [attribute_name, "width_modifier"]
            )
            return WidthModifier.from_dict(prior_dict)
        except ConfigException:
            logger.warning(
                f"No width modifier specified for class:\n\n"
                f"{cls.__name__}\n\n"
                f"and attribute:\n\n"
                f"{attribute_name}\n\n"
                f"Using default relative modifier of 0.5. See "
                f"https://pyautofit.readthedocs.io/en/latest/general/adding_a_model_component.html"
                f" for more information."
            )
            return RelativeWidthModifier(0.5)

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
    if inspect.isclass(obj) and issubclass(obj, WidthModifier)
}
