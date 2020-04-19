import configparser
from configparser import NoOptionError
import logging

from autofit import conf

logger = logging.getLogger(__name__)

def format_string_for_label(label: str) -> str:
    """
    Get the format for the label. Attempts to extract the key string associated with
    the dimension. Seems dodgy.

    Parameters
    ----------
    label
        A string label

    Returns
    -------
    format
        The format string (e.g {:.2f})
    """
    label_conf = conf.instance.label_format

    try:
        # noinspection PyProtectedMember
        for key, value in sorted(
            label_conf.parser._sections["format"].items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if key in label:
                return value
    except KeyError:
        pass
    raise configparser.NoSectionError(
        "Could not find format for label {} in config at path {}".format(
            label, label_conf.path
        )
    )


def label_and_label_string(label0, label1, whitespace):
    return label0 + label1.rjust(whitespace - len(label0) + len(label1))


def label_and_value_string(label, value, whitespace, format_string=None):
    format_str = format_string or format_string_for_label(label)
    value = format_str.format(value)
    return label + value.rjust(whitespace - len(label) + len(value))


def label_value_and_limits_string(
    label, value, lower_limit, upper_limit, whitespace, format_string=None
):
    format_str = format_string or format_string_for_label(label)
    value = format_str.format(value)
    upper_limit = format_str.format(upper_limit)
    lower_limit = format_str.format(lower_limit)
    value = value + " (" + lower_limit + ", " + upper_limit + ")"
    return label + value.rjust(whitespace - len(label) + len(value))


def label_value_and_unit_string(label, value, unit, whitespace, format_string=None):
    format_str = format_string or format_string_for_label(label)
    value = (format_str + " {}").format(value, unit)
    return label + value.rjust(whitespace - len(label) + len(value))


def output_list_of_strings_to_file(file, list_of_strings):
    file = open(file, "w")
    file.write("".join(list_of_strings))
    file.close()


def within_radius_label_value_and_unit_string(
    prefix, radius, unit_length, value, unit_value, whitespace
):
    label = prefix + "_within_{:.2f}_{}".format(radius, unit_length)
    return label_value_and_unit_string(
        label=label, value=value, unit=unit_value, whitespace=whitespace
    )

def param_labels_from_model(model) -> [str]:
    """A list of every parameter's label, used by *GetDist* for model estimation and visualization.

    The parameter labels are determined using the label.ini and label_format.ini config files."""

    paramnames_labels = []
    prior_class_dict = model.prior_class_dict
    prior_prior_model_dict = model.prior_prior_model_dict

    for prior_name, prior in model.prior_tuples_ordered_by_id:
        try:
            param_string = conf.instance.label.label(prior_name)
        except NoOptionError:
            logger.warning(
                f"No label provided for {prior_name}. Using prior name instead."
            )
            param_string = prior_name
        prior_model = prior_prior_model_dict[prior]
        cls = prior_class_dict[prior]
        cls_string = "{}{}".format(
            conf.instance.label.subscript(cls), prior_model.component_number + 1
        )
        param_label = "{}_{{\\mathrm{{{}}}}}".format(param_string, cls_string)
        paramnames_labels.append(param_label)

    return paramnames_labels
