"""
Per-``posterior_kind`` defaults and plotters (search-extensibility A4c, decision D14).

The ``AbstractMCMC`` / ``AbstractNest`` / ``AbstractMLE`` family bases carried three
defaults (initializer, clipper, auto-correlation settings) and one plot routine. They are
no longer in any public search's MRO; what they carried lives here, keyed by the
search's declared ``posterior_kind`` (``autofit.non_linear.search.capabilities``):

- ``CHAIN`` (Emcee, Zeus, BlackJAXNUTS): ``InitializerBall(0.49, 0.51)``, default
  ``AutoCorrelationsSettings``, the cornerpy corner plot once the chain is converged;
- ``WEIGHTED`` (Dynesty, Nautilus, NSS, SMC): ``InitializerPrior`` (nested samplers
  also reject ``InitializerParamBounds``), the anesthetic corner plot;
- ``POINT`` (BFGS, LBFGS, MultiStart, Drawer): ``InitializerBall(0.49, 0.51)``,
  ``ClipperNone``, the parameter / log-likelihood / figure-of-merit traces.

Each concrete search applies its kind's defaults in its own ``__init__``, never on
``NonLinearSearch``: the #1493 clipper tripwire requires that the clipper never reaches
the nested samplers. ``NonLinearSearch.plot_results`` plots through ``plotter_for``.
"""

import warnings
from typing import Optional

from autonerves import conf

from autofit.non_linear.search.capabilities import PosteriorKind


class PosteriorDefaults:
    kind: PosteriorKind = None
    plots_section: str = None

    def initializer_for(self, initializer=None):
        """The search's initializer: ``initializer``, or this kind's default."""
        return initializer if initializer is not None else self.default_initializer()

    def default_initializer(self):
        raise NotImplementedError

    def should_plot(self, name: str) -> bool:
        return conf.instance["visualize"]["plots_search"][self.plots_section][name]

    def plot_results(self, search, samples):
        raise NotImplementedError


class ChainDefaults(PosteriorDefaults):
    kind = PosteriorKind.CHAIN
    plots_section = "mcmc"

    def default_initializer(self):
        from autofit.non_linear.initializer import InitializerBall

        return InitializerBall(lower_limit=0.49, upper_limit=0.51)

    @staticmethod
    def auto_correlation_settings_for(auto_correlation_settings=None, **defaults):
        """
        The chain's auto-correlation settings: ``auto_correlation_settings``, or
        ``AutoCorrelationsSettings(**defaults)`` built at construction, so a run-time
        ``PYAUTO_TEST_MODE`` reaches its reduced ``check_size`` (the former import-time
        default instance did not).
        """
        from autofit.non_linear.search.mcmc.auto_correlations import (
            AutoCorrelationsSettings,
        )

        if auto_correlation_settings is None:
            return AutoCorrelationsSettings(**defaults)
        return auto_correlation_settings

    def plot_results(self, search, samples):
        from autofit.non_linear.plot import corner_cornerpy

        if not samples.pdf_converged:
            return

        if self.should_plot("corner_cornerpy"):
            corner_cornerpy(
                samples=samples,
                path=search.paths.image_path / "search",
                format="png",
            )


class WeightedDefaults(PosteriorDefaults):
    kind = PosteriorKind.WEIGHTED
    plots_section = "nest"

    def default_initializer(self):
        from autofit.non_linear.initializer import InitializerPrior

        return InitializerPrior()

    def nested_initializer_for(self, initializer=None):
        """
        A nested sampler's initializer: its initial live points are drawn from the prior,
        so ``InitializerParamBounds`` (a bounded start) is rejected.
        """
        from autofit.non_linear.initializer import InitializerParamBounds

        if isinstance(initializer, InitializerParamBounds):
            raise ValueError("InitializerParamBounds cannot be used for nested sampling")
        return self.initializer_for(initializer)

    def plot_results(self, search, samples):
        from autofit.non_linear.plot import corner_anesthetic

        if self.should_plot("corner_anesthetic"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corner_anesthetic(
                    samples=samples,
                    path=search.paths.image_path / "search",
                    format="png",
                )


class PointDefaults(PosteriorDefaults):
    kind = PosteriorKind.POINT
    plots_section = "mle"

    def default_initializer(self):
        from autofit.non_linear.initializer import InitializerBall

        return InitializerBall(lower_limit=0.49, upper_limit=0.51)

    @staticmethod
    def clipper_for(clipper=None):
        """
        The search's clipper, the no-op ``ClipperNone`` by default: enforcing prior
        support changes where a search converges, so it is opt-in.
        """
        from autofit.non_linear.clipper import ClipperNone

        return clipper or ClipperNone()

    def plot_results(self, search, samples):
        from autofit.non_linear.plot import (
            figure_of_merit_vs_iteration,
            log_likelihood_vs_iteration,
            subplot_parameters,
        )

        plot_path = search.paths.image_path / "search"

        if self.should_plot("subplot_parameters"):
            subplot_parameters(samples=samples, path=plot_path, format="png")
            subplot_parameters(
                samples=samples, use_log_y=True, path=plot_path, format="png"
            )
            subplot_parameters(
                samples=samples, use_last_50_percent=True, path=plot_path, format="png"
            )

        if self.should_plot("log_likelihood_vs_iteration"):
            log_likelihood_vs_iteration(samples=samples, path=plot_path, format="png")
            log_likelihood_vs_iteration(
                samples=samples, use_log_y=True, path=plot_path, format="png"
            )
            log_likelihood_vs_iteration(
                samples=samples, use_last_50_percent=True, path=plot_path, format="png"
            )

        # No-op unless the samples carry a ``fom_history`` (the auto-convergence
        # gradient searches); LBFGS / Drawer leave it absent. Default-on and
        # tolerant of a config predating this key (older workspaces that shadow
        # the ``mle`` plots section) so it never KeyErrors on an MLE fit.
        try:
            plot_figure_of_merit = self.should_plot("figure_of_merit_vs_iteration")
        except KeyError:
            plot_figure_of_merit = True
        if plot_figure_of_merit:
            figure_of_merit_vs_iteration(samples=samples, path=plot_path, format="png")


CHAIN = ChainDefaults()
WEIGHTED = WeightedDefaults()
POINT = PointDefaults()

_BY_KIND = {defaults.kind: defaults for defaults in (CHAIN, WEIGHTED, POINT)}


def defaults_for(kind) -> Optional[PosteriorDefaults]:
    """The defaults (and plotter) of a ``posterior_kind``, or ``None`` when undeclared."""
    if kind is None:
        return None
    return _BY_KIND[PosteriorKind(kind)]
