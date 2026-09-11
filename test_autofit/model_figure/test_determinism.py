"""
Determinism -- the same model must give the same figure, twice and everywhere.

Part of the epic's acceptance: a figure that is not byte-reproducible cannot be
diffed or committed as evidence.  The layout dict is the artefact (every
coordinate rounded to 4 dp), and the PNG bytes are checked directly.
"""

import hashlib
import itertools
import json
import subprocess
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")

import autofit as af

from test_autofit.graph_spec.lens_doubles import mge_model, simple_lens_model


def _reset_ids():
    af.ModelObject._ids = itertools.count()
    af.Prior._ids = itertools.count()


def _layout_json(model):
    return json.dumps(af.ModelPlotter(model).layout().to_dict())


def test_laying_out_the_same_model_twice_is_byte_identical():
    model = mge_model()

    assert _layout_json(model) == _layout_json(model)


def test_rebuilding_the_model_is_byte_identical():
    first = _layout_json(mge_model())

    _reset_ids()
    second = _layout_json(mge_model())

    assert first == second


def test_the_layout_is_identical_in_a_fresh_interpreter(tmp_path):
    """
    Text metrics come from a scratch Agg canvas, so they must not depend on
    anything the test session has already done to matplotlib.
    """
    expected = _layout_json(simple_lens_model())

    # Written to a file rather than stdout: importing the model doubles prints.
    target = tmp_path / "layout.json"
    script = textwrap.dedent(
        f"""
        import itertools, json, pathlib
        import matplotlib
        matplotlib.use("Agg")
        import autofit as af
        from test_autofit.graph_spec.lens_doubles import simple_lens_model
        af.ModelObject._ids = itertools.count()
        af.Prior._ids = itertools.count()
        pathlib.Path({str(target)!r}).write_text(
            json.dumps(af.ModelPlotter(simple_lens_model()).layout().to_dict())
        )
        """
    )
    subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    assert target.read_text() == expected


def test_png_bytes_are_identical_across_two_saves(tmp_path):
    model = simple_lens_model()

    af.ModelPlotter(model).figure(path=tmp_path, filename="first", format="png")
    af.ModelPlotter(model).figure(path=tmp_path, filename="second", format="png")

    first = hashlib.sha256((tmp_path / "first.png").read_bytes()).hexdigest()
    second = hashlib.sha256((tmp_path / "second.png").read_bytes()).hexdigest()
    assert first == second


def test_importing_the_model_figure_package_does_not_import_matplotlib():
    """
    ``af.ModelPlotter`` is exported from ``autofit/__init__.py``, so the drawing
    imports must all sit inside functions -- otherwise every ``import autofit``
    pays for matplotlib and ``test_autofit/graph_spec`` layer purity breaks.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, autofit.model_figure; "
            "print([m for m in sys.modules if m.split('.')[0] == 'matplotlib'])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "[]"
