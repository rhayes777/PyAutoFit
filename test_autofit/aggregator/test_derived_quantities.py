import pytest

from autofit import SearchOutput


@pytest.fixture(name="search_output")
def make_search_output(directory):
    return SearchOutput(directory / "search_output_derived")


def test(search_output):
    samples = search_output.samples
    assert samples.instance.fwhm == 10.0


def test_summary(search_output):
    summary = search_output.samples_summary
    assert summary.instance.fwhm == 15.0
