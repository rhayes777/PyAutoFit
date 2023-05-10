from pathlib import Path

import pytest

from autofit import SearchOutput


@pytest.fixture(name="child_analyses")
def make_child_analyses():
    directory = Path(__file__).parent / "search_output"
    search_output = SearchOutput(str(directory))
    return search_output.child_analyses


def test_child_analysis(child_analyses):
    assert len(child_analyses) == 2


def test_child_analysis_pickles(child_analyses):
    assert child_analyses[0].example == "hello world"
