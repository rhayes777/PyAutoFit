import logging
from pathlib import Path

import matplotlib.pyplot as plt

from autofit.graphical.expectation_propagation.history import EPHistory

logger = logging.getLogger(__name__)


class Visualise:
    def __init__(self, ep_history: EPHistory, output_path: Path):
        """
        Handles visualisation of expectation propagation optimisation.

        This includes plotting key metrics such as Evidence and KL Divergence
        which are expected to converge.

        Parameters
        ----------
        ep_history
            A history describing previous optimisations by factor
        output_path
            The path that plots are written to
        """
        self.ep_history = ep_history
        self.output_path = output_path

    def __call__(self):
        """
        Save a plot of Evidence and KL Divergence for the ep_history
        """
        fig, (evidence_plot, kl_plot) = plt.subplots(2)
        fig.suptitle("Evidence and KL Divergence")
        evidence_plot.plot(self.ep_history.evidences(), label="evidence")
        kl_plot.semilogy(self.ep_history.kl_divergences(), label="KL divergence")
        # for factor, factor_history in self.ep_history.items():
        #     evidence_plot.plot(
        #         factor_history.evidences, label=f"{factor.name} evidence"
        #     )
        #     kl_plot.plot(
        #         factor_history.kl_divergences, label=f"{factor.name} divergence"
        #     )
        evidence_plot.legend()
        kl_plot.legend()
        plt.savefig(str(self.output_path / "graph.png"))
