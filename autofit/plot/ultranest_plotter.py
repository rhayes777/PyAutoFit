from autofit.plot import SamplesPlotter

class UltraNestPlotter(SamplesPlotter):

    def cornerplot(self, **kwargs):

        from ultranest import plot

        plot.cornerplot(
            results=self.samples.results,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerplot")

    def runplot(self, **kwargs):

        from ultranest import plot

        try:
            plot.runplot(
                results=self.samples.results,
                **kwargs
            )
        except KeyError:
            pass

        self.output.to_figure(structure=None, auto_filename="runplot")

    def traceplot(self, **kwargs):

        from ultranest import plot

        try:
            plot.traceplot(
                results=self.samples.results,
                **kwargs
            )
        except KeyError:
            pass

        self.output.to_figure(structure=None, auto_filename="traceplot")