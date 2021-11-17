from typing import Optional

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples import Sample
from autofit.non_linear.samples.nest import NestSamples


class MultiNestSamples(NestSamples):

    def __init__(
            self,
            model: AbstractPriorModel,
            number_live_points: int,
            file_summary: str,
            file_weighted_samples: str,
            file_resume: str,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
    ):
        """
        Create a `Samples` object from this non-linear search's output files on the hard-disk and model.

        For MulitNest, this requires us to load:

            - The parameter samples, log likelihood values and weight_list from the multinest.txt file.
            - The total number of samples (e.g. accepted + rejected) from resume.dat.
            - The log evidence of the model-fit from the multinestsummary.txt file (if this is not yet estimated a
              value of -1.0e99 is used.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        """

        self.file_summary = file_summary
        self.file_weighted_samples = file_weighted_samples
        self.file_resume = file_resume

        self._number_live_points = number_live_points

        parameters = parameters_from_file_weighted_samples(
            file_weighted_samples=self.file_weighted_samples,
            prior_count=model.prior_count,
        )

        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector)) for vector in parameters
        ]

        log_likelihood_list = log_likelihood_list_from_file_weighted_samples(
            file_weighted_samples=self.file_weighted_samples
        )

        weight_list = weight_list_from_file_weighted_samples(
            file_weighted_samples=self.file_weighted_samples
        )

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        super().__init__(
            model=model,
            sample_list=sample_list,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
        )

    @property
    def number_live_points(self):
        return self._number_live_points

    @property
    def total_samples(self):
        return total_samples_from_file_resume(
            file_resume=self.file_resume
        )

    @property
    def log_evidence(self):
        return log_evidence_from_file_summary(
            file_summary=self.file_summary, prior_count=self.model.prior_count
        )


def parameters_from_file_weighted_samples(
        file_weighted_samples, prior_count
) -> [[float]]:
    """Open the file "multinest.txt" and extract the parameter values of every accepted live point as a list
    of lists."""
    weighted_samples = open(file_weighted_samples)

    total_samples = 0
    for line in weighted_samples:
        total_samples += 1

    weighted_samples.seek(0)

    parameters = []

    for line in range(total_samples):
        vector = []
        weighted_samples.read(56)
        for param in range(prior_count):
            vector.append(float(weighted_samples.read(28)))
        weighted_samples.readline()
        parameters.append(vector)

    weighted_samples.close()

    return parameters


def log_likelihood_list_from_file_weighted_samples(file_weighted_samples) -> [float]:
    """Open the file "multinest.txt" and extract the log likelihood values of every accepted live point as a list."""
    weighted_samples = open(file_weighted_samples)

    total_samples = 0
    for line in weighted_samples:
        total_samples += 1

    weighted_samples.seek(0)

    log_likelihood_list = []

    for line in range(total_samples):
        weighted_samples.read(28)
        log_likelihood_list.append(-0.5 * float(weighted_samples.read(28)))
        weighted_samples.readline()

    weighted_samples.close()

    return log_likelihood_list


def weight_list_from_file_weighted_samples(file_weighted_samples) -> [float]:
    """Open the file "multinest.txt" and extract the weight values of every accepted live point as a list."""
    weighted_samples = open(file_weighted_samples)

    total_samples = 0
    for line in weighted_samples:
        total_samples += 1

    weighted_samples.seek(0)

    log_likelihood_list = []

    for line in range(total_samples):
        weighted_samples.read(4)
        log_likelihood_list.append(float(weighted_samples.read(24)))
        weighted_samples.readline()

    weighted_samples.close()

    return log_likelihood_list


def total_samples_from_file_resume(file_resume):
    """Open the file "resume.dat" and extract the total number of samples of the MultiNest analysis
    (e.g. accepted + rejected)."""
    resume = open(file_resume)

    resume.seek(1)
    resume.read(19)
    total_samples = int(resume.read(8))
    resume.close()
    return total_samples


def log_evidence_from_file_summary(file_summary, prior_count):
    """Open the file "multinestsummary.txt" and extract the log evidence of the Multinest analysis.

    Early in the analysis this file may not yet have been created, in which case the log evidence estimate is
    unavailable and (would be unreliable anyway). In this case, a large negative value is returned."""

    try:

        with open(file_summary) as summary:

            summary.read(2 + 112 * prior_count)
            return float(summary.read(28))

    except FileNotFoundError:
        return -1.0e99
