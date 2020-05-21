import configparser
import logging

from autoconf import conf

logger = logging.getLogger(__name__)

class TextFormatter:
    def __init__(self, line_length=90, indent=4):
        self.dict = dict()
        self.line_length = line_length
        self.indent = indent

    def add_to_dict(self, path_item_tuple: tuple, info_dict: dict):
        path_tuple = path_item_tuple[0]
        key = path_tuple[0]
        if len(path_tuple) == 1:
            info_dict[key] = path_item_tuple[1]
        else:
            if key not in info_dict:
                info_dict[key] = dict()
            self.add_to_dict(
                (path_item_tuple[0][1:], path_item_tuple[1]), info_dict[key]
            )

    def add(self, path_item_tuple: tuple):
        self.add_to_dict(path_item_tuple, self.dict)

    def dict_to_list(self, info_dict, line_length):
        lines = []
        for key, value in info_dict.items():
            indent_string = self.indent * " "
            if isinstance(value, dict):
                sub_lines = self.dict_to_list(
                    value, line_length=line_length - self.indent
                )
                lines.append(key)
                for line in sub_lines:
                    lines.append(f"{indent_string}{line}")
            else:
                value_string = str(value)
                space_string = max((line_length - len(key)), 1) * " "
                lines.append(f"{key}{space_string}{value_string}")
        return lines

    @property
    def list(self):
        return self.dict_to_list(self.dict, line_length=self.line_length)

    @property
    def text(self):
        return "\n".join(self.list)


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


