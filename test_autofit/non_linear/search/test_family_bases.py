"""
The family bases out of the MRO (search-extensibility A4c, decision D14).

``AbstractMCMC`` / ``AbstractNest`` / ``AbstractMLE`` are deprecated thin subclasses of
``NonLinearSearch``: subclassing one warns, and a subclass keeps the former defaults.
The public searches apply their ``posterior_kind`` defaults in their own ``__init__``
and plot through the per-kind plotter.
"""

import json
import warnings
from pathlib import Path

import pytest

import autofit as af
from autofit.non_linear.initializer import InitializerBall, InitializerPrior
from autofit.non_linear.samples.mcmc import SamplesMCMC
from autofit.non_linear.samples.nest import SamplesNest
from autofit.non_linear.search import posterior_kind
from autofit.non_linear.search.mcmc.abstract_mcmc import AbstractMCMC
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings
from autofit.non_linear.search.mle.abstract_mle import AbstractMLE
from autofit.non_linear.search.nest.abstract_nest import AbstractNest
from autonerves.dictable import get_class_path

from test_autofit.non_linear.search.conformance_roster import searches_under_test


def _subclass(base):
    """Define a minimal external search on a deprecated base, as a user would."""

    class External(base):
        __identifier_fields__ = ()

        def _fit(self, model, analysis):
            raise NotImplementedError

    return External


@pytest.mark.parametrize("base", [AbstractMCMC, AbstractNest, AbstractMLE])
def test_subclassing_a_deprecated_family_base_warns(base):
    with pytest.warns(FutureWarning, match=f"subclasses {base.__name__}"):
        _subclass(base)


@pytest.mark.parametrize("base", [AbstractMCMC, AbstractNest, AbstractMLE])
def test_deprecated_bases_are_distinct_classes(base):
    assert base is not af.NonLinearSearch
    assert issubclass(base, af.NonLinearSearch)


def test_deprecated_mcmc_keeps_its_defaults():
    with pytest.warns(FutureWarning):
        External = _subclass(AbstractMCMC)

    search = External()

    assert isinstance(search.initializer, InitializerBall)
    assert (search.initializer.lower_limit, search.initializer.upper_limit) == (
        0.49,
        0.51,
    )
    assert isinstance(search.auto_correlation_settings, AutoCorrelationsSettings)
    assert External.samples_cls is SamplesMCMC
    assert not hasattr(search, "clipper")


def test_deprecated_nest_keeps_its_defaults():
    with pytest.warns(FutureWarning):
        External = _subclass(AbstractNest)

    assert isinstance(External().initializer, InitializerPrior)
    assert External.samples_cls is SamplesNest
    with pytest.raises(ValueError, match="nested sampling"):
        External(initializer=af.InitializerParamBounds({}))


def test_deprecated_mle_keeps_its_defaults():
    with pytest.warns(FutureWarning):
        External = _subclass(AbstractMLE)

    search = External()

    assert isinstance(search.initializer, InitializerBall)
    assert isinstance(search.clipper, af.ClipperNone)


def test_public_searches_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        for entry in searches_under_test():
            if not entry.missing_requirements():
                entry.resolve()()


@pytest.mark.parametrize(
    "cls, initializer",
    [
        (af.Emcee, InitializerBall),
        (af.Zeus, InitializerBall),
        (af.BlackJAXNUTS, InitializerBall),
        (af.SMC, InitializerPrior),
        (af.DynestyStatic, InitializerPrior),
        (af.Nautilus, InitializerPrior),
        (af.Drawer, InitializerBall),
        (af.BFGS, InitializerBall),
        (af.MultiStartAdam, InitializerBall),
    ],
)
def test_posterior_kind_defaults_applied(cls, initializer):
    assert type(cls().initializer) is initializer


@pytest.mark.parametrize("cls", [af.DynestyStatic, af.DynestyDynamic, af.Nautilus])
def test_nested_samplers_still_reject_param_bounds(cls):
    with pytest.raises(ValueError, match="nested sampling"):
        cls(initializer=af.InitializerParamBounds({}))


def test_smc_is_weighted_and_takes_no_auto_correlation_settings():
    search = af.SMC()

    assert str(af.SMC.posterior_kind) == "weighted"
    assert af.SMC.produces_evidence is True
    assert not hasattr(search, "auto_correlation_settings")


def test_old_smc_search_json_with_auto_correlation_settings_loads(caplog):
    golden = json.loads(
        (Path(__file__).parent / "seed_golden.json").read_text()
    )
    old = golden["SMC"]["main_search_json"]
    assert "auto_correlation_settings" in old["arguments"]

    loaded = af.from_dict(old)

    assert isinstance(loaded, af.SMC)
    assert "unknown keyword argument" not in caplog.text


# `samples_info["class_path"]` is the search's `samples_cls`; main 8437a3b76's values.
SAMPLES_CLASS_PATHS = {
    "Emcee": "autofit.non_linear.samples.mcmc.SamplesMCMC",
    "Zeus": "autofit.non_linear.samples.mcmc.SamplesMCMC",
    "BlackJAXNUTS": "autofit.non_linear.samples.mcmc.SamplesMCMC",
    "SMC": "autofit.non_linear.search.mcmc.blackjax.smc.samples.SamplesSMC",
    "DynestyStatic": "autofit.non_linear.samples.nest.SamplesNest",
    "DynestyDynamic": "autofit.non_linear.samples.nest.SamplesNest",
    "Nautilus": "autofit.non_linear.samples.nest.SamplesNest",
    "NSS": "autofit.non_linear.search.nest.nss.samples.NSSamples",
    "Drawer": "autofit.non_linear.samples.samples.Samples",
    "BFGS": "autofit.non_linear.samples.samples.Samples",
    "LBFGS": "autofit.non_linear.samples.samples.Samples",
    "MultiStartAdam": "autofit.non_linear.samples.samples.Samples",
    "MultiStartADABelief": "autofit.non_linear.samples.samples.Samples",
    "MultiStartLion": "autofit.non_linear.samples.samples.Samples",
    "MultiStartProdigy": "autofit.non_linear.samples.samples.Samples",
}


@pytest.mark.parametrize("entry", searches_under_test(), ids=lambda e: e.name)
def test_samples_class_path_unchanged(entry):
    assert get_class_path(entry.resolve().samples_cls) == SAMPLES_CLASS_PATHS[entry.name]


@pytest.mark.parametrize(
    "cls, kind",
    [(af.Emcee, posterior_kind.CHAIN), (af.SMC, posterior_kind.WEIGHTED),
     (af.Nautilus, posterior_kind.WEIGHTED), (af.BFGS, posterior_kind.POINT)],
)
def test_plot_results_dispatches_on_posterior_kind(cls, kind, monkeypatch):
    calls = []
    monkeypatch.setattr(
        type(kind), "plot_results", lambda self, search, samples: calls.append(self)
    )

    cls().plot_results(samples=None)

    assert calls == [kind]


def test_search_without_a_posterior_kind_plots_nothing_of_its_own():
    with pytest.raises(NotImplementedError):
        af.m.MockSearch().plot_results(samples=None)


# Adversary review 07, finding 4: positional constructor compatibility.


def test_drawer_positional_slots_are_unchanged():
    """`Drawer`'s explicit `clipper` is appended, so main's positional calls bind as before."""
    from autofit.non_linear.clipper import ClipperNone

    drawer = af.Drawer(None, None, None, 50, None, 12, 34, True)

    assert drawer.total_draws == 50
    assert drawer.iterations_per_full_update == 12
    assert drawer.iterations_per_quick_update == 34
    assert drawer.silence is True
    assert isinstance(drawer.clipper, ClipperNone)


def test_drawer_clipper_still_accepted_by_keyword_and_serialized():
    from autofit.non_linear.clipper import ClipperNone
    from autonerves.dictable import get_arguments

    assert "clipper" in get_arguments(af.Drawer)
    assert isinstance(af.Drawer(clipper=None).clipper, ClipperNone)
    assert "clipper" in af.to_dict(af.Drawer())["arguments"]


def _smc_positional_prefix():
    # name, path_prefix, unique_tag, num_particles, kernel, num_mcmc_steps,
    # num_integration_steps, target_ess, step_size, whiten_inflate, max_smc_steps,
    # batch_size, seed, initializer, inverse_mass_matrix
    return (None, None, None, 256, "mala", 5, 8, 0.5, None, 2.0, 200, 0, None, None, None)


def test_smc_old_positional_auto_correlation_slot_binds_as_before():
    pytest.importorskip("blackjax")
    settings = AutoCorrelationsSettings(check_for_convergence=False)

    with pytest.warns(FutureWarning, match="auto_correlation_settings"):
        search = af.SMC(*_smc_positional_prefix(), settings, 3, 7, 1, True)

    assert search.iterations_per_quick_update == 3
    assert search.iterations_per_full_update == 7
    assert search.number_of_cores == 1
    assert search.silence is True
    assert "auto_correlation_settings" not in af.to_dict(search)["arguments"]


def test_smc_too_many_positional_arguments_raise():
    pytest.importorskip("blackjax")

    with pytest.raises(TypeError, match="deprecated positional"):
        af.SMC(*_smc_positional_prefix(), None, 3, 7, 1, True, None, "extra")


def test_smc_keyword_construction_does_not_warn():
    pytest.importorskip("blackjax")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        search = af.SMC(iterations_per_quick_update=3, silence=True)

    assert search.iterations_per_quick_update == 3
    assert search.silence is True
