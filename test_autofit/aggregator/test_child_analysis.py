from pathlib import Path

import pytest

from autofit import SearchOutput
from autofit.aggregator import Aggregator


@pytest.fixture(name="directory")
def make_directory():
    return Path(__file__).parent


@pytest.fixture(name="search_output")
def make_search_output(directory):
    return SearchOutput(directory / "search_output")


@pytest.fixture(name="child_analyses")
def make_child_analyses(search_output):
    return search_output.child_analyses


def test_child_analysis(child_analyses):
    assert len(child_analyses) == 2


def test_child_analysis_pickles(child_analyses):
    assert child_analyses[0].example == "hello world"


def test_child_analysis_values(directory):
    aggregator = Aggregator(directory)

    assert list(aggregator.child_values("example")) == [["hello world", "hello world"]]
