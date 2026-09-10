import pytest

import autofit as af
from autofit.non_linear.fitness import Fitness

# A quick update is a convenience render: it draws the current best fit and
# rewrites `model.results`. It must never be able to end a search.
#
# `manage_quick_update` used to call `model.instance_from_vector` unguarded. The
# max-likelihood vector a sampler is carrying is not guaranteed to map to a
# physical instance -- PyAutoGalaxy's `ModelParameterException` (which subclasses
# both `ValueError` and `af.exc.FitException`) is raised straight out of a profile
# constructor for e.g. an out-of-disk `ell_comps`. That exception unwound through
# the likelihood call the sampler was driving and killed a 36 h Nautilus run at
# its first quick update (PyAutoFit#1567).
#
# The model class below reproduces exactly that shape: a constructor that raises
# a ValueError/FitException subclass for a parameter outside its domain.


class _OutOfDomain(ValueError, af.exc.FitException):
    """Mirrors PyAutoGalaxy's `ModelParameterException` inheritance."""


class _BoundedGaussian:
    def __init__(
        self,
        centre: float = 0.0,
        normalization: float = 1.0,
        sigma: float = 1.0,
    ):
        if sigma <= 0.0:
            raise _OutOfDomain(
                f"sigma must be positive, got {sigma}."
            )

        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma


class RecordingPaths:
    """Captures the rendered result info instead of writing it to disk."""

    def __init__(self):
        self.results = []

    def output_model_results(self, result_info):
        self.results.append(result_info)


class RecordingAnalysis(af.Analysis):
    """Records every quick-update render instead of drawing one."""

    def __init__(self, raises=None):
        super().__init__()
        self.instances = []
        self.raises = raises

    def perform_quick_update(self, paths, instance):
        self.instances.append(instance)

        if self.raises is not None:
            raise self.raises


def _model():
    # Priors are given explicitly so the throwaway test class needs no entry in
    # the prior config. Their ranges are irrelevant here: `instance_from_vector`
    # takes physical values, so the vector alone decides whether the constructor
    # raises.
    return af.Model(
        _BoundedGaussian,
        centre=af.UniformPrior(lower_limit=0.0, upper_limit=100.0),
        normalization=af.UniformPrior(lower_limit=0.0, upper_limit=100.0),
        sigma=af.UniformPrior(lower_limit=-10.0, upper_limit=10.0),
    )


def _fitness(analysis, iterations_per_quick_update=1):
    fitness = Fitness(
        model=_model(),
        analysis=analysis,
        iterations_per_quick_update=iterations_per_quick_update,
    )
    fitness.paths = RecordingPaths()
    return fitness


def test__an_invalid_max_lh_instance_skips_the_visual_instead_of_killing_the_search(
    caplog,
):
    analysis = RecordingAnalysis()
    fitness = _fitness(analysis)

    with caplog.at_level("INFO", logger="autofit.non_linear.fitness"):
        # sigma = -1.0 is outside the model's domain, so `instance_from_vector`
        # raises inside the update body. Before the guard this propagated out of
        # `manage_quick_update` and ended the search.
        fitness.manage_quick_update(
            parameters=[50.0, 25.0, -1.0], log_likelihood=-10.0
        )

    # No render happened...
    assert analysis.instances == []

    # ...but the counter still reset, or every later evaluation would re-fire
    # the update and re-log the skip.
    assert fitness.quick_update_count == 0

    assert "Quick update skipped" in caplog.text

    # The results text is formatted from the parameter vector alone, so it is
    # still written even when no instance could be built.
    assert len(fitness.paths.results) == 1
    assert "centre" in fitness.paths.results[0]


def test__a_valid_max_lh_instance_still_renders_the_visual(caplog):
    analysis = RecordingAnalysis()
    fitness = _fitness(analysis)

    with caplog.at_level("INFO", logger="autofit.non_linear.fitness"):
        fitness.manage_quick_update(
            parameters=[50.0, 25.0, 10.0], log_likelihood=-10.0
        )

    assert len(analysis.instances) == 1
    assert analysis.instances[0].sigma == 10.0

    assert "Quick update skipped" not in caplog.text

    assert fitness.quick_update_count == 0
    assert len(fitness.paths.results) == 1


@pytest.mark.parametrize("live_visual_update", [False, True])
def test__quick_updates_preserve_the_live_notebook_display(monkeypatch, live_visual_update):
    cleared = []
    monkeypatch.setattr("IPython.display.clear_output", lambda **kwargs: cleared.append(kwargs))
    fitness = _fitness(RecordingAnalysis())
    fitness.live_visual_update = live_visual_update

    for likelihood in [-10.0, -9.0]:
        fitness.manage_quick_update(
            parameters=[50.0, 25.0, 10.0], log_likelihood=likelihood
        )

    # Text-only updates can clear their old output. Live updates must retain
    # the image's display_id target across subsequent likelihood evaluations.
    assert len(cleared) == (0 if live_visual_update else 2)


def test__a_visual_that_raises_is_swallowed_like_the_background_worker_does(caplog):
    # The background quick-update worker already swallows `Exception` around the
    # render (`BackgroundQuickUpdate._process_pending`). The synchronous path
    # must not be the more fragile of the two.
    analysis = RecordingAnalysis(raises=RuntimeError("plotting blew up"))
    fitness = _fitness(analysis)

    with caplog.at_level("INFO", logger="autofit.non_linear.fitness"):
        fitness.manage_quick_update(
            parameters=[50.0, 25.0, 10.0], log_likelihood=-10.0
        )

    assert len(analysis.instances) == 1
    assert "ignored" in caplog.text

    assert fitness.quick_update_count == 0
    assert len(fitness.paths.results) == 1
