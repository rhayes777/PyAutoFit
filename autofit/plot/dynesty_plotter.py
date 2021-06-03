from autoconf import conf
from autofit.plot import SamplesPlotter

from dynesty import plotting as dyplot


class DynestyPlotter(SamplesPlotter):

    def boundplot(self, **kwargs):

        if conf.instance["general"]["test"]["test_mode"]:
            return None

        dyplot.boundplot(
            results=self.samples.results,
            labels=self.model.parameter_labels_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="boundplot")
        self.mat_plot_1d.figure.close()

    def cornerbound(self, **kwargs):

        if conf.instance["general"]["test"]["test_mode"]:
            return None

        dyplot.cornerbound(
            results=self.samples.results,
            labels=self.model.parameter_labels_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerbound")
        self.mat_plot_1d.figure.close()

    def cornerplot(self, **kwargs):

        if conf.instance["general"]["test"]["test_mode"]:
            return None

        dyplot.cornerplot(
            results=self.samples.results,
            labels=self.model.parameter_labels_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerplot")
        self.mat_plot_1d.figure.close()

    def cornerpoints(self, **kwargs):

        if conf.instance["general"]["test"]["test_mode"]:
            return None

        try:
            dyplot.cornerpoints(
                results=self.samples.results,
                labels=self.model.parameter_labels_latex,
                **kwargs
            )

            self.output.to_figure(structure=None, auto_filename="cornerpoints")
        except ValueError:
            pass

        self.mat_plot_1d.figure.close()

    def runplot(self, **kwargs):

        if conf.instance["general"]["test"]["test_mode"]:
            return None

        try:
            dyplot.runplot(
                results=self.samples.results,
                **kwargs
            )
        except ValueError:
            pass

        self.output.to_figure(structure=None, auto_filename="runplot")
        self.mat_plot_1d.figure.close()

    def traceplot(self, **kwargs):

        if conf.instance["general"]["test"]["test_mode"]:
            return None

        dyplot.traceplot(
            results=self.samples.results,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="traceplot")
        self.mat_plot_1d.figure.close()