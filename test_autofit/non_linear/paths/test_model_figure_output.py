"""
The per-search ``model.png`` hook in ``DirectoryPaths._save_model_info``.

The figure is opt-in: ``output.yaml``'s ``model_figure`` key is read strictly,
so a config which has never heard of the key writes no figure even though its
``default:`` is ``true``. And a figure bug must never kill a fit, so a renderer
which raises leaves ``model.info`` written and the exception swallowed into the
log.
"""

from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

import autofit as af
from autonerves import conf

from autofit.non_linear.paths.directory import _model_figure_enabled


@contextmanager
def model_figure_key(value):
    """
    Set ``output.yaml``'s ``model_figure`` key for the duration of a test, or
    remove it entirely when ``value`` is ``None`` (the "config has never heard
    of this key" case, which must still be off).
    """
    output = conf.instance["output"]
    missing = object()
    original = output["model_figure"] if "model_figure" in output else missing

    if value is None:
        if original is not missing:
            del output["model_figure"]
    else:
        output["model_figure"] = value

    try:
        yield
    finally:
        if original is missing:
            if "model_figure" in output:
                del output["model_figure"]
        else:
            output["model_figure"] = original


@contextmanager
def output_path(tmp_path):
    """Point ``conf.instance.output_path`` at a temporary directory."""
    original = conf.instance.output_path
    conf.instance.output_path = str(tmp_path)
    try:
        yield
    finally:
        conf.instance.output_path = original


@pytest.fixture(name="model")
def make_model():
    return af.Model(af.ex.Gaussian)


def _paths_for(model):
    paths = af.DirectoryPaths(name="model_figure")
    paths.model = model
    paths.search = af.DynestyStatic()
    return paths


def _written(paths) -> Path:
    return Path(paths.output_path)


class TestModelFigureEnabled:
    def test__key_true(self):
        with model_figure_key(True):
            assert _model_figure_enabled() is True

    def test__key_false(self):
        with model_figure_key(False):
            assert _model_figure_enabled() is False

    def test__key_absent_is_off_even_though_default_is_true(self):
        """
        ``autonerves.output.should_output`` would return ``default`` here, which
        is ``true``. The hook must not: an absent key means off.
        """
        with model_figure_key(None):
            assert conf.instance["output"]["default"] is True
            assert _model_figure_enabled() is False


class TestModelPngOutput:
    def test__key_true__model_png_written_beside_model_info(self, model, tmp_path):
        with output_path(tmp_path), model_figure_key(True):
            paths = _paths_for(model)
            paths._save_model_info(model=model)

            directory = _written(paths)
            assert (directory / "model.info").exists()
            assert (directory / "model.png").exists()
            assert (directory / "model.png").stat().st_size > 0

    def test__key_false__no_model_png(self, model, tmp_path):
        with output_path(tmp_path), model_figure_key(False):
            paths = _paths_for(model)
            paths._save_model_info(model=model)

            directory = _written(paths)
            assert (directory / "model.info").exists()
            assert not (directory / "model.png").exists()

    def test__key_absent__no_model_png(self, model, tmp_path):
        with output_path(tmp_path), model_figure_key(None):
            paths = _paths_for(model)
            paths._save_model_info(model=model)

            directory = _written(paths)
            assert (directory / "model.info").exists()
            assert not (directory / "model.png").exists()

    def test__skip_visualization__no_model_png(self, model, tmp_path, monkeypatch):
        monkeypatch.setenv("PYAUTO_SKIP_VISUALIZATION", "1")

        with output_path(tmp_path), model_figure_key(True):
            paths = _paths_for(model)
            paths._save_model_info(model=model)

            directory = _written(paths)
            assert (directory / "model.info").exists()
            assert not (directory / "model.png").exists()

    def test__renderer_raises__model_info_still_written_and_nothing_propagates(
        self, model, tmp_path, monkeypatch
    ):
        """
        A figure bug must never kill a fit.
        """
        from autofit.model_figure import ModelPlotter

        def raise_(*args, **kwargs):
            raise RuntimeError("the renderer blew up")

        monkeypatch.setattr(ModelPlotter, "figure", raise_)

        with output_path(tmp_path), model_figure_key(True):
            paths = _paths_for(model)
            paths._save_model_info(model=model)

            directory = _written(paths)
            assert (directory / "model.info").exists()
            assert not (directory / "model.png").exists()

    def test__renderer_failure_is_logged(self, model, tmp_path, monkeypatch, caplog):
        import logging

        from autofit.model_figure import ModelPlotter

        def raise_(*args, **kwargs):
            raise RuntimeError("the renderer blew up")

        monkeypatch.setattr(ModelPlotter, "figure", raise_)

        with output_path(tmp_path), model_figure_key(True):
            paths = _paths_for(model)
            with caplog.at_level(logging.INFO):
                paths._save_model_info(model=model)

        assert "model.png not written" in caplog.text
