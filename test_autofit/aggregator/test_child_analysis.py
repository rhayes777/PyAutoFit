from pathlib import Path

from autofit import SearchOutput


def test_child_analysis():
    directory = Path(__file__).parent / "search_output"
    search_output = SearchOutput(str(directory))

    assert len(search_output.child_analyses) == 2
