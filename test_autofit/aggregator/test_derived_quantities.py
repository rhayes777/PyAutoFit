import pytest

from autofit import SearchOutput


@pytest.fixture(name="search_output")
def make_search_output(directory):
    return SearchOutput(directory / "search_output_derived")


def test(search_output):
    samples = search_output.samples
    assert samples.instance.fwhm == 10.0
