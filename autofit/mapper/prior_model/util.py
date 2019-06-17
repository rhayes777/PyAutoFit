import re

from autofit.mapper.prior import AttributeNameValue


def tuple_name(attribute_name):
    """
    Extract the name of a tuple attribute from the name of one of its components,
    e.g. centre_0 -> origin

    Parameters
    ----------
    attribute_name: str
        The name of an attribute which is a component of a tuple

    Returns
    -------
    tuple_name: str
        The name of the tuple of which the attribute is a member
    """
    return "_".join(attribute_name.split("_")[:-1])


def is_tuple_like_attribute_name(attribute_name):
    """
    Determine if a string matches the pattern "{attribute_name}_#", that is if it
    seems to be a tuple.

    Parameters
    ----------
    attribute_name: str
        The name of some attribute that may refer to a tuple.

    Returns
    -------
    is_tuple_like: bool
        True iff the attribute name looks like that which refers to a tuple.
    """
    pattern = re.compile("^[a-zA-Z_0-9]*_[0-9]$")
    return pattern.match(attribute_name)


class PriorModelNameValue(AttributeNameValue):
    @property
    def prior_model(self):
        return self.value
