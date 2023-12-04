from typing import List, Tuple

from autofit.non_linear.grid.grid_list import GridList, as_grid_list
from autofit.non_linear.samples.interface import SamplesInterface

# noinspection PyTypeChecker
class SensitivityResult:
    def __init__(
            self,
            samples: List[SamplesInterface],
            perturb_samples: List[SamplesInterface],
            shape : Tuple[int, ...]
    ):
        """
        The result of a sensitivity mapping

        Parameters
        ----------
        results
            The results of each sensitivity job.
        shape
            The shape of the sensitivity mapping grid.
        """
        self.samples = GridList(samples, shape)
        self.perturb_samples = GridList(perturb_samples, shape)
        self.shape = shape

    def __getitem__(self, item):
        return self.samples[item]

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    @property
    @as_grid_list
    def log_evidences_base(self) -> GridList:
        """
        The log evidences of the base model for each sensitivity fit
        """
        return [sample.log_evidence for sample in self.samples]

    @property
    @as_grid_list
    def log_evidences_perturbed(self) -> GridList:
        """
        The log evidences of the perturbed model for each sensitivity fit
        """
        return [sample.log_evidence for sample in self.perturb_samples]

    @property
    @as_grid_list
    def log_evidence_differences(self) -> GridList:
        """
        The log evidence differences between the base and perturbed models
        """
        return [
            log_evidence_perturbed - log_evidence_base for
            log_evidence_perturbed, log_evidence_base in
            zip(self.log_evidences_perturbed, self.log_evidences_base)
        ]

    @property
    @as_grid_list
    def log_likelihoods_base(self) -> GridList:
        """
        The log likelihoods of the base model for each sensitivity fit
        """
        return [sample.log_likelihood for sample in self.samples]

    @property
    @as_grid_list
    def log_likelihoods_perturbed(self) -> GridList:
        """
        The log likelihoods of the perturbed model for each sensitivity fit
        """
        return [sample.log_likelihood for sample in self.perturb_samples]

    @property
    @as_grid_list
    def log_likelihood_differences(self) -> GridList:
        """
        The log likelihood differences between the base and perturbed models
        """
        return [
            log_likelihood_perturbed - log_likelihood_base for
            log_likelihood_perturbed, log_likelihood_base in
            zip(self.log_likelihoods_perturbed, self.log_likelihoods_base)
        ]

    def figure_of_merits(
        self, use_log_evidences: bool,
    ) -> GridList:
        """
        Convenience method to get either the log likelihoods difference or log evidence difference of the grid search.

        Parameters
        ----------
        use_log_evidences
            If true, the log evidences are returned, otherwise the log likelihoods are returned.
        """

        if use_log_evidences:
            return self.log_evidence_differences
        return self.log_likelihood_differences



