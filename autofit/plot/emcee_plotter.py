from autofit.plot import SamplesPlotter

import corner

class EmceePlotter(SamplesPlotter):

    def corner(self, **kwargs):

        corner.corner(
            xs=self.samples.parameters,
            weights=self.samples.weights,
            labels=self.model.parameter_labels_latex,
            **kwargs
        )

        self.output.to_figure(structure=None)