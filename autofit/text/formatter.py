import logging
from typing import Tuple

from autoconf import conf
from autofit.tools.util import open_

logger = logging.getLogger(__name__)


class FormatNode:
    def __init__(self):
        self._dict = dict()
        self.value = None

    def __getitem__(self, item):
        if item not in self._dict:
            self._dict[item] = FormatNode()
        return self._dict[item]

    def __len__(self):
        return len(self._dict)

    def items(self):
        return self._dict.items()


class TextFormatter:
    def __init__(self, line_length=90, indent=4):
        self.dict = FormatNode()
        self.line_length = line_length
        self.indent = indent

    def add_to_dict(self, path: Tuple[str, ...], value: str, info_dict: FormatNode):
        key = path[0]
        node = info_dict[key]
        if len(path) == 1:
            node.value = value
        else:
            self.add_to_dict(
                path[1:], value, node
            )

    def add(self, path: Tuple[str, ...], value):
        self.add_to_dict(path, value, self.dict)

    def dict_to_list(self, info_dict, line_length):
        lines = []
        for key, value in info_dict.items():
            indent_string = self.indent * " "
            if value.value is not None:
                value_string = str(value.value)
                space_string = max((line_length - len(key)), 1) * " "
                lines.append(f"{key}{space_string}{value_string}")

            if len(value) > 0:
                sub_lines = self.dict_to_list(
                    value, line_length=line_length - self.indent
                )
                if value.value is None:
                    lines.append(key)
                for line in sub_lines:
                    lines.append(f"{indent_string}{line}")
        return lines

    @property
    def list(self):
        return self.dict_to_list(self.dict, line_length=self.line_length)

    @property
    def text(self):
        return "\n".join(self.list)


def format_string_for_parameter_name(parameter_name: str) -> str:
    """
    Get the format for the label. Attempts to extract the key string associated with
    the dimension. Seems dodgy.

    Parameters
    ----------
    parameter_name
        A string label

    Returns
    -------
    format
        The format string (e.g {:.2f})
    """
    label_conf = conf.instance["notation"]["label_format"]

    try:
        # noinspection PyProtectedMember
        for key, value in sorted(
                label_conf["format"].items(),
                key=lambda item: len(item[0]),
                reverse=True,
        ):
            if key in parameter_name:
                return value
    except KeyError:
        pass

    logger.debug(
        "Could not find an entry for the parameter {} in the label_format.ini config at path {}".format(
            parameter_name, conf.instance.paths
        )
    )

    return "{:.4f}"


def convert_name_to_label(parameter_name, name_to_label):
    if not name_to_label:
        return parameter_name

    label_conf = conf.instance["notation"]["label"]

    try:
        return label_conf["label"][parameter_name]
    except KeyError:
        logger.debug(
            "Could not find an entry for the parameter {} in the label_format.iniconfig at paths {}".format(
                parameter_name, conf.instance.paths
            )
        )
        return parameter_name[0]


def add_whitespace(str0, str1, whitespace):
    return f"{str0}{str1.rjust(whitespace - len(str0) + len(str1))}"


def value_result_string_from(
        parameter_name, value, values_at_sigma=None, unit=None, format_string=None
):
    format_str = format_string or format_string_for_parameter_name(parameter_name)
    value = format_str.format(value)

    if unit is not None:
        unit = f" {unit}"
    else:
        unit = ""

    if values_at_sigma is None:
        return f"{value}{unit}"
    else:
        lower_value_at_sigma = format_str.format(values_at_sigma[0])
        upper_value_at_sigma = format_str.format(values_at_sigma[1])
        return f"{value} ({lower_value_at_sigma}, {upper_value_at_sigma}){unit}"


def parameter_result_latex_from(
        parameter_name,
        value,
        errors=None,
        superscript="",
        unit=None,
        format_string=None,
        name_to_label=False,
        include_name=True,
        include_quickmath=False,
):
    format_str = format_string or format_string_for_parameter_name(parameter_name)
    value = format_str.format(value)

    name = convert_name_to_label(
        parameter_name=parameter_name, name_to_label=name_to_label
    )

    if unit is not None:
        unit = f" {unit}"
    else:
        unit = ""

    if not superscript:
        superscript = ""
    else:
        superscript = f"^{{\\rm{{{superscript}}}}}"

    if errors is None:

        if include_name:
            parameter_result_latex = f"{name}{superscript} = {value}{unit}"
        else:
            parameter_result_latex = f"{value}{unit}"

    else:

        lower_value_at_sigma = format_str.format(errors[0])
        upper_value_at_sigma = format_str.format(errors[1])

        parameter_result = f"{value}^{{+{upper_value_at_sigma}}}_{{-{lower_value_at_sigma}}}{unit}"

        if include_name:
            parameter_result_latex = f"{name}{superscript} = {value}^{{+{upper_value_at_sigma}}}_{{-{lower_value_at_sigma}}}{unit}"
        else:
            parameter_result_latex = parameter_result

    if "e" in format_str:

        psplit = parameter_result.split("e")

        parameter_result_latex = (
            f"" 
            f"{psplit[0]}" 
            f"{psplit[1][3:]}"
            f"{psplit[2][3:]}" 
            f"{psplit[3][-1]}" 
            f" \\times 10^{{{int(psplit[1][1:3])}}}"
        )

    if not include_quickmath:
        return f"{parameter_result_latex} & "
    return f"${parameter_result_latex}$ & "


def output_list_of_strings_to_file(file, list_of_strings):
    with open_(file, "w") as f:
        f.write("".join(list_of_strings))
