import configparser

from autofit import conf


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
    label = label.split("_within_")[0]
    label = label.split("_at_")[0]
    try:
        return conf.instance.label_format.get("format", label)
    except configparser.NoOptionError:
        label = "_".join(label.split("_")[1:])
        if len(label) == 0:
            raise
        return format_string_for_label(label)


def label_and_label_string(label0, label1, whitespace):
    return label0 + label1.rjust(whitespace - len(label0) + len(label1))


def label_and_value_string(
        label,
        value,
        whitespace,
        format_string=None
):
    format_str = format_string or format_string_for_label(label)
    value = format_str.format(value)
    return label + value.rjust(whitespace - len(label) + len(value))


def label_value_and_limits_string(
        label,
        value,
        lower_limit,
        upper_limit,
        whitespace,
        format_string=None
):
    format_str = format_string or format_string_for_label(label)
    value = format_str.format(value)
    upper_limit = format_str.format(upper_limit)
    lower_limit = format_str.format(lower_limit)
    value = value + ' (' + lower_limit + ', ' + upper_limit + ')'
    return label + value.rjust(whitespace - len(label) + len(value))


def label_value_and_unit_string(
        label,
        value,
        unit,
        whitespace,
        format_string=None
):
    format_str = format_string or format_string_for_label(label)
    value = (format_str + ' {}').format(value, unit)
    return label + value.rjust(whitespace - len(label) + len(value))


def output_list_of_strings_to_file(file, list_of_strings):
    file = open(file, 'w')
    file.write(''.join(list_of_strings))
    file.close()
