"""
The capability matrix's catalogue column (``docs/_generate_searches.py``): a search
with a card in the autofit_inference catalogue links to it at the pinned commit, with a
line anchor; a search without one reads "no catalogue card yet". No network is used.

Skipped when the ``docs`` directory is absent (an installed-package test run).
"""

import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest

from autofit.non_linear.search import registry

GENERATOR = Path(__file__).resolve().parents[1] / "docs" / "_generate_searches.py"

pytestmark = pytest.mark.skipif(
    not GENERATOR.exists(), reason="docs/ is not part of this checkout"
)

COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(name="generator")
def make_generator():
    spec = importlib.util.spec_from_file_location("_generate_searches", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="catalogue_text")
def make_catalogue_text():
    # A wave-2 style section repeats a "searches" key first, so the anchor must come
    # from the top-level ``tasks`` block, not the first match in the file.
    catalogue = {
        "schema": "autofit-inference-search-catalogue",
        "early": {"searches": {"Nautilus": {"status": "decoy"}}},
        "tasks": {
            "point_map": {
                "label": "point/MAP",
                "searches": {"LBFGS": {"status": "measured", "legs": []}},
            },
            "evidence": {
                "label": "evidence",
                "searches": {"Nautilus": {"status": "deferred"}},
            },
        },
    }
    return json.dumps(catalogue, indent=2)


def _line_of(text, needle):
    lines = text.splitlines()
    return max(index for index, line in enumerate(lines) if needle in line) + 1


def test_cards_keyed_by_search_name_with_line_anchor(generator, catalogue_text):
    cards = generator.catalogue_cards(catalogue_text)

    assert set(cards) == {"LBFGS", "Nautilus"}
    assert cards["LBFGS"]["label"] == "point/MAP"
    assert cards["Nautilus"]["status"] == "deferred"
    assert cards["Nautilus"]["line"] == _line_of(catalogue_text, '"Nautilus": {')
    assert catalogue_text.splitlines()[cards["LBFGS"]["line"] - 1].strip() == (
        '"LBFGS": {'
    )


def test_present_and_absent_cells(generator, catalogue_text):
    cards = generator.catalogue_cards(catalogue_text)

    present = generator._card_cell({"name": "LBFGS"}, cards, COMMIT)
    absent = generator._card_cell({"name": "Emcee"}, cards, COMMIT)

    line = cards["LBFGS"]["line"]
    assert present == (
        "`point/MAP <https://github.com/PyAutoLabs/autofit_inference/blob/"
        f"{COMMIT}/catalogue/search_catalogue.json#L{line}>`__ (measured)"
    )
    assert absent == generator.NO_CARD == "no catalogue card yet"


def test_matrix_has_one_catalogue_cell_per_search_and_records_the_pin(
    generator, catalogue_text
):
    text = generator.searches_index_rst(
        registry.manifest(), generator.catalogue_cards(catalogue_text), COMMIT
    )
    capabilities = text.split("Capabilities\n------------")[1].split("Objectives\n")[0]

    assert f"PyAutoLabs/autofit_inference@{COMMIT}" in text
    assert capabilities.count(f"/blob/{COMMIT}/catalogue/search_catalogue.json#L") == 2
    assert capabilities.count("no catalogue card yet") == len(registry.entries()) - 2


def test_unreadable_catalogue_raises(generator):
    with pytest.raises(generator.CatalogueError):
        generator.catalogue_cards("not json")


def test_http_error_is_an_error_not_a_skip(generator, monkeypatch, capsys):
    def not_found(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(generator.urllib.request, "urlopen", not_found)

    with pytest.raises(generator.CatalogueError):
        generator.load_catalogue_text()
    assert generator.main(["--check"]) == 1
    assert "404" in capsys.readouterr().err


def test_offline_check_skips_only_the_matrix(generator, monkeypatch, capsys):
    def offline(*args, **kwargs):
        raise generator.CatalogueUnavailable("no network")

    monkeypatch.setattr(generator, "load_catalogue_text", offline)

    assert generator.main(["--check"]) == 0
    captured = capsys.readouterr()
    assert "skipped docs/searches/index.rst" in captured.err
    assert "2 generated search pages are up to date." in captured.out

    assert generator.main(["--check", "--require-catalogue"]) == 1
