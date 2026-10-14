"""
The template smoke (search-extensibility A5): a toy search written from
``autofit/non_linear/search/_template/search.py`` (``template_toy_search.ToyHillClimb``)
passes both layers of the search conformance suite -- layer (i), metadata and
serialization (``test_conformance.py``), and layer (ii), backend execution
(``test_conformance_backend.py``) -- through the very same test functions, with the
toy's own golden rows.
"""

import inspect

import pytest

from test_autofit.non_linear.search import conformance_roster as roster
from test_autofit.non_linear.search import test_conformance as layer_i
from test_autofit.non_linear.search import test_conformance_backend as layer_ii

TOY = roster.SearchEntry(
    "ToyHillClimb",
    "test_autofit.non_linear.search.template_toy_search.ToyHillClimb",
    "mle",
    jax_use="none",
)

TOY_GOLDEN = roster.Golden(
    identifier="d2250e3156c424efc1ff5c9f00267506",
    identifier_fields=("n_steps", "step_size"),
    constructor_arguments={
        "initializer",
        "iterations_per_full_update",
        "iterations_per_quick_update",
        "kwargs",
        "n_steps",
        "name",
        "number_of_cores",
        "path_prefix",
        "session",
        "silence",
        "step_size",
        "unique_tag",
    },
)


@pytest.fixture(autouse=True)
def toy_rows(monkeypatch):
    """
    The toy's rows in the golden tables both layers read by search name.
    """
    monkeypatch.setitem(layer_i.GOLDEN, TOY.name, TOY_GOLDEN)
    monkeypatch.setitem(layer_ii.SEARCH_INTERNAL_FILES, TOY.name, layer_ii._DILL)
    monkeypatch.setitem(
        layer_ii.SAMPLES_INFO_KEYS,
        TOY.name,
        layer_ii._CORE_KEYS | {"total_steps", "n_accepted"},
    )
    monkeypatch.setitem(layer_ii.SAMPLES_CLASS, TOY.name, "Samples")


def test_the_toy_is_the_template_renamed():
    template = roster.resolve_class_path(
        "autofit.non_linear.search._template.search.TemplateSearch"
    )

    assert inspect.getsource(TOY.resolve()) == inspect.getsource(template).replace(
        "TemplateSearch", "ToyHillClimb"
    )


def test_the_toy_is_at_most_250_lines():
    assert len(inspect.getsource(inspect.getmodule(TOY.resolve())).splitlines()) <= 250


LAYER_I = [
    layer_i.test_default_construction,
    layer_i.test_family_base,
    layer_i.test_identifier_fields_golden,
    layer_i.test_constructor_argument_set,
    layer_i.test_search_json_round_trip,
]


@pytest.mark.parametrize("case", LAYER_I, ids=lambda case: case.__name__)
def test_layer_i(case):
    case(TOY)


def test_layer_i_identifier(monkeypatch):
    layer_i.test_identifier_golden(TOY, monkeypatch)


def test_layer_i_config_unchanged_by_construction(tmp_path):
    fixture = layer_i.make_search_internal_off.__wrapped__(tmp_path)
    next(fixture)
    try:
        layer_i.test_config_unchanged_by_construction(TOY, None)
    finally:
        next(fixture, None)


@pytest.fixture(name="toy_run", scope="module")
def make_toy_run(tmp_path_factory):
    return layer_ii._execute(TOY, tmp_path_factory.mktemp("backend_toy"))


LAYER_II = [
    layer_ii.test_backend_runs_in_reduced_test_mode,
    layer_ii.test_directory_paths_output_file_set,
    layer_ii.test_completed_path_resumes_without_rerunning_the_backend,
    layer_ii.test_null_paths_fit_returns_a_result,
    layer_ii.test_null_paths_samples_from_none_raises_rather_than_falling_back,
    layer_ii.test_restored_output_samples_from_none_loads_the_internal_state,
    layer_ii.test_samples_info_key_set,
]


@pytest.mark.parametrize("case", LAYER_II, ids=lambda case: case.__name__)
def test_layer_ii(case, toy_run):
    case(TOY, lambda entry: toy_run)


@pytest.mark.parametrize("error_type", [FileNotFoundError, NotImplementedError])
def test_layer_ii_samples_from_falls_back(error_type, toy_run, monkeypatch, caplog):
    layer_ii.test_samples_from_falls_back_to_paths_samples(
        TOY, error_type, lambda entry: toy_run, monkeypatch, caplog
    )
