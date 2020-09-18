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


def test__add_whitespace_between_strings():
    str0 = af.text.formatter.add_whitespace(str0="param0", str1="mass", whitespace=10)

    assert str0 == "param0    mass"

    str0 = af.text.formatter.add_whitespace(str0="param0", str1="mass", whitespace=20)

    assert str0 == "param0              mass"


def test__parameter_name_and_value_string():
    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param0", value=2.0, whitespace=10
    )
    assert str0 == "param0    2.00"

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param11", value=3.00, whitespace=20
    )

    assert str0 == "param11             3.0000"

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param12", value=3.00, whitespace=15
    )

    assert str0 == "param12        3.00e+00"

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param0", value=3.00, whitespace=15, name_to_label=True
    )

    assert str0 == "p0             3.00"


def test__parameter_name_value_and_limits_string():
    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param0", value=2.0, values_at_sigma=(1.5, 2.5), whitespace=10
    )
    assert str0 == "param0    2.00 (1.50, 2.50)"

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param11",
        value=3.00,
        values_at_sigma=(0.0001, 40000.0),
        whitespace=20,
    )

    assert str0 == "param11             3.0000 (0.0001, 40000.0000)"

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param12",
        value=3.00,
        values_at_sigma=(1.0, 500.0),
        whitespace=15,
    )

    assert str0 == "param12        3.00e+00 (1.00e+00, 5.00e+02)"

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param0",
        value=2.0,
        values_at_sigma=(1.5, 2.5),
        whitespace=10,
        name_to_label=True,
    )
    assert str0 == "p0        2.00 (1.50, 2.50)"


def test__parameter_name_with_subscript_value_and_limits_string():
    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param0",
        subscript="a",
        value=2.0,
        values_at_sigma=(1.5, 2.5),
        whitespace=10,
    )

    assert str0 == "param0_a  2.00 (1.50, 2.50)"

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param11",
        value=3.00,
        values_at_sigma=(0.0001, 40000.0),
        subscript="d",
        whitespace=20,
    )

    assert str0 == "param11_d           3.0000 (0.0001, 40000.0000)"

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param12",
        value=3.00,
        values_at_sigma=(1.0, 500.0),
        subscript="e",
        whitespace=15,
    )

    assert str0 == "param12_e      3.00e+00 (1.00e+00, 5.00e+02)"

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param0",
        subscript="a",
        value=2.0,
        values_at_sigma=(1.5, 2.5),
        whitespace=10,
        name_to_label=True,
    )

    assert str0 == "p0_a      2.00 (1.50, 2.50)"


def test__parameter_name_value_and_unit_string():
    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param0", value=2.0, unit="arcsec", whitespace=10
    )

    assert str0 == "param0    2.00 arcsec"

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param11", value=3.00, unit="kg", whitespace=20
    )

    assert str0 == "param11             3.0000 kg"

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param12", value=3.00, unit="kgs", whitespace=15
    )

    assert str0 == "param12        3.00e+00 kgs"

    str0 = af.text.formatter.parameter_result_string_from(
        parameter_name="param0",
        value=2.0,
        unit="arcsec",
        whitespace=10,
        name_to_label=True,
    )

    assert str0 == "p0        2.00 arcsec"


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
        af.text.formatter.format_string_for_parameter_name("mass_value") == "mass_value"
    )


def test_substring():
    assert (
        af.text.formatter.format_string_for_parameter_name("einstein_radius")
        == "radius_value"
    )
    assert (
        af.text.formatter.format_string_for_parameter_name("mass_value_something")
        == "mass_value"
    )
