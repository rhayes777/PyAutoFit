from autofit.plot import SamplesPlotter

from dynesty import plotting as dyplot


class DynestyPlotter(SamplesPlotter):

    def boundplot(self, **kwargs):

        dyplot.boundplot(
            results=self.samples.results,
            labels=self.model.parameter_labels_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="boundplot")

    def cornerbound(self, **kwargs):

        dyplot.cornerbound(
            results=self.samples.results,
            labels=self.model.parameter_labels_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerbound")

    def cornerplot(self, **kwargs):

        dyplot.cornerplot(
            results=self.samples.results,
            labels=self.model.parameter_labels_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerplot")

    def cornerpoints(self, **kwargs):

        dyplot.cornerpoints(
            results=self.samples.results,
            labels=self.model.parameter_labels_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerpoints")

    def runplot(self, **kwargs):

        dyplot.runplot(
            results=self.samples.results,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="runplot")

    def traceplot(self, **kwargs):

        dyplot.traceplot(
            results=self.samples.results,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="traceplot")