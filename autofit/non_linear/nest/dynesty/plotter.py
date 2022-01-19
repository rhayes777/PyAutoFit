from dynesty import plotting as dyplot

from autoconf import conf
from autofit.plot import SamplesPlotter



class DynestyPlotter(SamplesPlotter):

    def boundplot(self, **kwargs):

        if conf.instance["general"]["test"]["test_mode"]:
            return None

        dyplot.boundplot(
            results=self.samples.results_internal,
            labels=self.model.parameter_labels_with_superscripts_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="boundplot")
        self.close()

    def cornerbound(self, **kwargs):

        if conf.instance["general"]["test"]["test_mode"]:
            return None

        dyplot.cornerbound(
            results=self.samples.results_internal,
            labels=self.model.parameter_labels_with_superscripts_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerbound")
        self.close()

    def cornerplot(self, **kwargs):

        if conf.instance["general"]["test"]["test_mode"]:
            return None

        dyplot.cornerplot(
            results=self.samples.results_internal,
            labels=self.model.parameter_labels_with_superscripts_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerplot")
        self.close()

    def cornerpoints(self, **kwargs):

        if conf.instance["general"]["test"]["test_mode"]:
            return None

        try:
            dyplot.cornerpoints(
                results=self.samples.results_internal,
                labels=self.model.parameter_labels_with_superscripts_latex,
                **kwargs
            )

            self.output.to_figure(structure=None, auto_filename="cornerpoints")
        except ValueError:
            pass

        self.close()

    def runplot(self, **kwargs):

        if conf.instance["general"]["test"]["test_mode"]:
            return None

        try:
            dyplot.runplot(
                results=self.samples.results_internal,
                **kwargs
            )
        except ValueError:
            pass

        self.output.to_figure(structure=None, auto_filename="runplot")
        self.close()

    def traceplot(self, **kwargs):

        if conf.instance["general"]["test"]["test_mode"]:
            return None

        dyplot.traceplot(
            results=self.samples.results_internal,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="traceplot")
        self.close()