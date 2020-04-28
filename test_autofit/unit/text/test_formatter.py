import os
import shutil

import pytest

import autofit as af
from autofit.text import formatter
from autofit import conf

from test_autofit.mock import MockClassNLOx4

directory = os.path.dirname(os.path.realpath(__file__))

text_path = "{}/files/text/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    conf.instance = conf.Config(
        config_path="{}/files/config/text".format(directory)
    )


def test__label_and_label_string():
    string0 = formatter.label_and_label_string(
        label0="param0", label1="mass", whitespace=10
    )
    string1 = formatter.label_and_label_string(
        label0="param00", label1="mass0", whitespace=10
    )
    string2 = formatter.label_and_label_string(
        label0="param000", label1="mass111", whitespace=10
    )

    assert string0 == "param0    mass"
    assert string1 == "param00   mass0"
    assert string2 == "param000  mass111"

    string0 = formatter.label_and_label_string(
        label0="param0", label1="mass", whitespace=20
    )

    assert string0 == "param0              mass"


def test__label_and_value_string():
    string0 = formatter.label_and_value_string(
        label="param0", value=2.0, whitespace=10
    )
    string1 = formatter.label_and_value_string(
        label="param00", value=2.0, whitespace=10
    )
    string2 = formatter.label_and_value_string(
        label="param000", value=2.0, whitespace=10
    )

    assert string0 == "param0    2.00"
    assert string1 == "param00   2.00"
    assert string2 == "param000  2.00"

    string = formatter.label_and_value_string(
        label="param11", value=3.00, whitespace=20
    )

    assert string == "param11             3.0000"

    string = formatter.label_and_value_string(
        label="param12", value=3.00, whitespace=15
    )

    assert string == "param12        3.00e+00"


def test__label_value_and_limits_string():
    string0 = formatter.label_value_and_limits_string(
        label="param0", value=2.0, upper_limit=2.5, lower_limit=1.5, whitespace=10
    )
    string1 = formatter.label_value_and_limits_string(
        label="param00", value=2.0, upper_limit=2.7, lower_limit=1.3, whitespace=10
    )
    string2 = formatter.label_value_and_limits_string(
        label="param000", value=2.0, upper_limit=2.9, lower_limit=1.1, whitespace=10
    )

    assert string0 == "param0    2.00 (1.50, 2.50)"
    assert string1 == "param00   2.00 (1.30, 2.70)"
    assert string2 == "param000  2.00 (1.10, 2.90)"

    string = formatter.label_value_and_limits_string(
        label="param11",
        value=3.00,
        upper_limit=40000.0,
        lower_limit=0.0001,
        whitespace=20,
    )

    assert string == "param11             3.0000 (0.0001, 40000.0000)"

    string = formatter.label_value_and_limits_string(
        label="param12", value=3.00, upper_limit=500.0, lower_limit=1.0, whitespace=15
    )

    assert string == "param12        3.00e+00 (1.00e+00, 5.00e+02)"


def test__label_value_and_unit_string():
    string0 = formatter.label_value_and_unit_string(
        label="param0", value=2.0, unit="arcsec", whitespace=10
    )
    string1 = formatter.label_value_and_unit_string(
        label="param00", value=2.0, unit="mass", whitespace=10
    )
    string2 = formatter.label_value_and_unit_string(
        label="param000", value=2.0, unit="kg", whitespace=10
    )

    assert string0 == "param0    2.00 arcsec"
    assert string1 == "param00   2.00 mass"
    assert string2 == "param000  2.00 kg"

    string = formatter.label_value_and_unit_string(
        label="param11", value=3.00, unit="kg", whitespace=20
    )

    assert string == "param11             3.0000 kg"

    string = formatter.label_value_and_unit_string(
        label="param12", value=3.00, unit="kgs", whitespace=15
    )

    assert string == "param12        3.00e+00 kgs"


def test__output_list_of_strings_to_file():
    if os.path.exists(text_path):
        shutil.rmtree(text_path)

    os.mkdir(text_path)

    results = ["hi\n", "hello"]
    formatter.output_list_of_strings_to_file(
        file=text_path + "model.results", list_of_strings=results
    )

    file = open(text_path + "model.results", "r")

    assert file.readlines() == ["hi\n", "hello"]


def test__within_radius_label_value_and_unit_string():
    string0 = formatter.within_radius_label_value_and_unit_string(
        prefix="mass",
        radius=1.0,
        unit_length="arcsec",
        value=30.0,
        unit_value="solMass",
        whitespace=40,
    )

    string1 = formatter.within_radius_label_value_and_unit_string(
        prefix="mass",
        radius=1.0,
        unit_length="arcsec",
        value=30.0,
        unit_value="solMass",
        whitespace=35,
    )

    string2 = formatter.within_radius_label_value_and_unit_string(
        prefix="mass",
        radius=1.0,
        unit_length="arcsec",
        value=30.0,
        unit_value="solMass",
        whitespace=30,
    )

    assert string0 == "mass_within_1.00_arcsec                 mass_value 30.0"
    assert string1 == "mass_within_1.00_arcsec            mass_value 30.0"
    assert string2 == "mass_within_1.00_arcsec       mass_value 30.0"

    string = formatter.within_radius_label_value_and_unit_string(
        prefix="mass",
        radius=1.0,
        unit_length="arcsec2",
        value=40.0,
        unit_value="solMass2",
        whitespace=40,
    )

    assert string == "mass_within_1.00_arcsec2                mass_value 40.0"


def test_string():
    assert formatter.format_string_for_label("radius_value") == "radius_value"
    assert formatter.format_string_for_label("mass_value") == "mass_value"


def test_substring():
    assert formatter.format_string_for_label("einstein_radius") == "radius_value"
    assert formatter.format_string_for_label("mass_value_something") == "mass_value"


def test_param_labels_labels():

    model = af.PriorModel(MockClassNLOx4)

    param_labels = formatter.param_labels_from_model(model=model)

    assert param_labels == [
        r"x4p0_{\mathrm{a1}}",
        r"x4p1_{\mathrm{a1}}",
        r"x4p2_{\mathrm{a1}}",
        r"x4p3_{\mathrm{a1}}",
    ]
