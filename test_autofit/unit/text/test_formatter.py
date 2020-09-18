import os
import shutil

import pytest

import autofit as af
from autoconf import conf

directory = os.path.dirname(os.path.realpath(__file__))

text_path = "{}/files/text/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    conf.instance = conf.Config(config_path="{}/files/config/text".format(directory))


def test__parameter_result_string__basic_inputs():

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param0", value=2.0,
    )
    assert str0 == "param0 2.00"

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param11", value=3.00,
    )

    assert str0 == "param11 3.0000"


def test__parameter_result_string__name_to_label_changes_parameter_name_to_label():

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param0", value=3.00, name_to_label=True
    )

    assert str0 == "p0 3.00"


def test__parameter_result_string__with_limits_string():

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param0", value=2.0, values_at_sigma=(1.5, 2.5),
    )
    assert str0 == "param0 2.00 (1.50, 2.50)"


def test__parameter_result_string__include_subscript():
    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param0",
        subscript="a",
        value=2.0,
        values_at_sigma=(1.5, 2.5),
    )

    assert str0 == "param0_a 2.00 (1.50, 2.50)"


def test__parameter_result_string__include_unit():
    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param0", value=2.0, unit="arcsec",
    )

    assert str0 == "param0 2.00 arcsec"

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param11", value=3.00, unit="kg",
    )

    assert str0 == "param11 3.0000 kg"


def test__parameter_result_latex():

    str0 = af.text.formatter.parameter_result_latex_from(
        parameter_name="param0", value=2.0,
    )
    assert str0 == r"param0 = 2.00 & "

    str0 = af.text.formatter.parameter_result_latex_from(
        parameter_name="param0", value=2.0, values_at_sigma=(0.1, 0.2),
    )
    assert str0 == r"param0 = 2.00^{+0.20}_{-0.10} & "

    str0 = af.text.formatter.parameter_result_latex_from(
        parameter_name="param0", value=3.00, subscript="a", values_at_sigma=(0.1, 0.2)
    )

    assert str0 == r"param0_{\mathrm{a}} = 3.00^{+0.20}_{-0.10} & "

    str0 = af.text.formatter.parameter_result_latex_from(
        parameter_name="param0", value=3.00, subscript="a", name_to_label=True
    )

    assert str0 == r"p0_{\mathrm{a}} = 3.00 & "

    str0 = af.text.formatter.parameter_result_latex_from(
        parameter_name="param0", value=3.00, subscript="a", unit="kg"
    )

    assert str0 == r"param0_{\mathrm{a}} = 3.00 kg & "


def test__output_list_of_strings_to_file():
    if os.path.exists(text_path):
        shutil.rmtree(text_path)

    os.mkdir(text_path)

    results = ["hi\n", "hello"]
    af.text.formatter.output_list_of_strings_to_file(
        file=text_path + "model.results", list_of_strings=results
    )

    file = open(text_path + "model.results", "r")

    assert file.readlines() == ["hi\n", "hello"]


def test_string():
    assert (
        af.text.formatter.format_string_for_parameter_name("radius_value")
        == "radius_value"
    )
    assert (
        af.text.formatter.format_string_for_parameter_name("mass_value") == "{:.2f}"
    )


def test_substring():
    assert (
        af.text.formatter.format_string_for_parameter_name("einstein_radius")
        == "radius_value"
    )
    assert (
        af.text.formatter.format_string_for_parameter_name("mass_value_something")
        == "{:.2f}"
    )

