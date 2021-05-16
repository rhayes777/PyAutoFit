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
        self.mat_plot_1d.figure.close()

    def cornerbound(self, **kwargs):

        dyplot.cornerbound(
            results=self.samples.results,
            labels=self.model.parameter_labels_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerbound")
        self.mat_plot_1d.figure.close()

    def cornerplot(self, **kwargs):

        dyplot.cornerplot(
            results=self.samples.results,
            labels=self.model.parameter_labels_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerplot")
        self.mat_plot_1d.figure.close()

    def cornerpoints(self, **kwargs):

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

        dyplot.traceplot(
            results=self.samples.results,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="traceplot")
        self.mat_plot_1d.figure.close()