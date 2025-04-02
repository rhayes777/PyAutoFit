import csv

from pathlib import Path

from autofit.aggregator.summary.aggregate_csv import AggregateCSV, ValueType

import pytest


@pytest.fixture
def output_path():
    path = Path("/tmp/summary.csv")
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def summary(aggregator):
    return AggregateCSV(aggregator)


@pytest.fixture
def load_output(output_path):
    def _load_output():
        with open(output_path) as f:
            return list(csv.DictReader(f))

    return _load_output


def test_writes(output_path, summary):
    summary.save(output_path)

    with open(output_path) as f:
        dicts = list(csv.DictReader(f))

    assert dicts[0]["id"] is not None
    assert dicts[1]["id"] is not None


def test_add_label_colum(
    output_path,
    summary,
    load_output,
):
    summary.add_label_column("label", ["a", "b"])
    summary.save(output_path)

    dicts = load_output()

    assert dicts[0]["label"] == "a"
    assert dicts[1]["label"] == "b"


def test_add_column(
    output_path,
    summary,
    load_output,
):
    summary.add_column("galaxies.lens.bulge.centre.centre_0")
    summary.save(output_path)

    dicts = load_output()

    assert dicts[0]["galaxies_lens_bulge_centre_centre_0"] == "-1.0" or "-5.0"
    assert dicts[1]["galaxies_lens_bulge_centre_centre_0"] == "-5.0" or "-1.0"


def test_use_max_log_likelihood(
    output_path,
    summary,
    load_output,
):
    summary.add_column(
        "galaxies.lens.bulge.centre.centre_0",
        value_types=[ValueType.MaxLogLikelihood],
    )
    summary.save(output_path)

    dicts = load_output()

    assert dicts[0]["galaxies_lens_bulge_centre_centre_0_max_lh"] == "-1.0" or "-5.0"
    assert dicts[1]["galaxies_lens_bulge_centre_centre_0_max_lh"] == "-5.0" or "-1.0"


def test_add_named_column(
    output_path,
    summary,
    load_output,
):
    summary.add_column(
        "galaxies.lens.bulge.centre.centre_0",
        name="centre_0",
    )
    summary.save(output_path)

    dicts = load_output()

    assert dicts[0]["centre_0"] == "-1.0" or "-5.0"
    assert dicts[1]["centre_0"] == "-5.0" or "-1.0"


def test_add_latent_column(
    output_path,
    summary,
    load_output,
):
    summary.add_column(
        "latent.value",
    )
    summary.save(output_path)

    dicts = load_output()

    assert dicts[0]["latent_value"] == "1.0" or "2.0"
    assert dicts[1]["latent_value"] == "2.0" or "1.0"


def test_computed_column(
    output_path,
    summary,
    load_output,
):
    def compute(samples):
        return 1

    summary.add_computed_column(
        "computed",
        compute,
    )
    summary.save(output_path)

    dicts = load_output()

    assert dicts[0]["computed"] == "1"


def test_dict_computed_column(
    output_path,
    summary,
    load_output,
):
    def compute(samples):
        return {"a": 1, "b": 2}

    summary.add_computed_column(
        "computed",
        compute,
    )
    summary.save(output_path)

    dicts = load_output()

    first = dicts[0]
    assert first["computed_a"] == "1"
    assert first["computed_b"] == "2"


def test_values_at_sigma(
    output_path,
    summary,
    load_output,
):
    summary.add_column(
        "galaxies.lens.bulge.centre.centre_0",
        value_types=[ValueType.ValuesAt1Sigma],
    )
    summary.save(output_path)

    dicts = load_output()

    print(dicts)
