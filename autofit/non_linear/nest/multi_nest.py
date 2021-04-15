import os
import shutil
from os import path

from autoconf import conf
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear import abstract_search
from autofit.non_linear import result as res
from autofit.non_linear.log import logger
from autofit.non_linear.nest import abstract_nest
from autofit.non_linear.samples import NestSamples, Sample


class MultiNest(abstract_nest.AbstractNest):

    def __init__(
            self,
            name=None,
            path_prefix=None,
            prior_passer=None,
            **kwargs
    ):
        """
        A MultiNest non-linear search.

        For a full description of MultiNest and its Python wrapper PyMultiNest, checkout its Github and documentation
        webpages:

        https://github.com/JohannesBuchner/MultiNest
        https://github.com/JohannesBuchner/PyMultiNest
        http://johannesbuchner.github.io/PyMultiNest/index.html#

        Parameters
        ----------
        name : str
            The name of the search, controlling the last folder results are output.
        path_prefix : str
            The path of folders prefixing the name folder where results are output.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            prior_passer=prior_passer,
            **kwargs
        )

        logger.debug("Creating MultiNest NLO")

    class Fitness(abstract_nest.AbstractNest.Fitness):

        def __init__(
                self,
                paths,
                model,
                analysis,
                samples_from_model,
                stagger_resampling_likelihood,
                log_likelihood_cap=None,
                pool_ids=None
        ):

            super().__init__(
                model=model,
                analysis=analysis,
                samples_from_model=samples_from_model,
                stagger_resampling_likelihood=stagger_resampling_likelihood,
                log_likelihood_cap=log_likelihood_cap,
                pool_ids=pool_ids,
                paths=paths
            )

            should_update_sym = conf.instance["non_linear"]["nest"]["MultiNest"]["updates"]["should_update_sym"]

            self.should_update_sym = abstract_search.IntervalCounter(should_update_sym)

        def fit_instance(self, instance):

            if self.should_update_sym():
                self.paths.copy_from_sym()

            log_likelihood = self.analysis.log_likelihood_function(instance=instance)

            if self.log_likelihood_cap is not None:
                if log_likelihood > self.log_likelihood_cap:
                    log_likelihood = self.log_likelihood_cap

            if log_likelihood > self.max_log_likelihood:
                self.max_log_likelihood = log_likelihood

            return log_likelihood

    def _fit(self, model: AbstractPriorModel, analysis, log_likelihood_cap=None) -> res.Result:
        """
        Fit a model using MultiNest and the Analysis class which contains the data and returns the log likelihood from
        instances of the model, which the `NonLinearSearch` seeks to maximize.

        Parameters
        ----------
        model : ModelMapper
            The model which generates instances for different points in parameter space.
        analysis : Analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the `NonLinearSearch` maximizes.

        Returns
        -------
        A result object comprising the Samples object that includes the maximum log likelihood instance and full
        set of accepted ssamples of the fit.
        """

        # noinspection PyUnusedLocal
        def prior(cube, ndim, nparams):
            # NEVER EVER REFACTOR THIS LINE! Haha.

            phys_cube = model.vector_from_unit_vector(unit_vector=cube)

            for i in range(len(phys_cube)):
                cube[i] = phys_cube[i]

            return cube

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis
        )

        import pymultinest

        logger.info("Beginning MultiNest non-linear search. ")

        pymultinest.run(
            fitness_function,
            prior,
            model.prior_count,
            outputfiles_basename="{}/multinest".format(self.paths.path),
            verbose=not self.silence,
            **self.config_dict
        )
        self.copy_from_sym()

    def samples_via_sampler_from_model(self, model: AbstractPriorModel):
        """Create a `Samples` object from this non-linear search's output files on the hard-disk and model.

        For MulitNest, this requires us to load:

            - The parameter samples, log likelihood values and weights from the multinest.txt file.
            - The total number of samples (e.g. accepted + rejected) from resume.dat.
            - The log evidence of the model-fit from the multinestsummary.txt file (if this is not yet estimated a
              value of -1.0e99 is used.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        """

        parameters = parameters_from_file_weighted_samples(
            file_weighted_samples=self.file_weighted_samples,
            prior_count=model.prior_count,
        )

        log_priors = [
            sum(model.log_priors_from_vector(vector=vector)) for vector in parameters
        ]

        log_likelihoods = log_likelihoods_from_file_weighted_samples(
            file_weighted_samples=self.file_weighted_samples
        )

        weights = weights_from_file_weighted_samples(
            file_weighted_samples=self.file_weighted_samples
        )

        total_samples = total_samples_from_file_resume(
            file_resume=self.file_resume
        )

        log_evidence = log_evidence_from_file_summary(
            file_summary=self.file_summary, prior_count=model.prior_count
        )

        return NestSamples(
            model=model,
            samples=Sample.from_lists(
                parameters=parameters,
                log_likelihoods=log_likelihoods,
                log_priors=log_priors,
                weights=weights,
                model=model
            ),
            total_samples=total_samples,
            log_evidence=log_evidence,
            number_live_points=self.n_live_points,
            time=self.timer.time
        )

    @property
    def file_summary(self) -> str:
        return path.join(self.paths.samples_path, "multinestsummary.txt")

    @property
    def file_weighted_samples(self):
        return path.join(self.paths.samples_path, "multinest.txt")

    @property
    def file_resume(self) -> str:
        return path.join(self.paths.samples_path, "multinestresume.dat")

    def copy_from_sym(self):
        """
        Copy files from the sym-linked search folder to the samples folder.
        """

        src_files = os.listdir(self.paths.path)
        for file_name in src_files:
            full_file_name = path.join(self.paths.path, file_name)
            if path.isfile(full_file_name):
                shutil.copy(full_file_name, self.paths.samples_path)


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


def log_likelihoods_from_file_weighted_samples(file_weighted_samples) -> [float]:
    """Open the file "multinest.txt" and extract the log likelihood values of every accepted live point as a list."""
    weighted_samples = open(file_weighted_samples)

    total_samples = 0
    for line in weighted_samples:
        total_samples += 1

    weighted_samples.seek(0)

    log_likelihoods = []

    for line in range(total_samples):
        weighted_samples.read(28)
        log_likelihoods.append(-0.5 * float(weighted_samples.read(28)))
        weighted_samples.readline()

    weighted_samples.close()

    return log_likelihoods


def weights_from_file_weighted_samples(file_weighted_samples) -> [float]:
    """Open the file "multinest.txt" and extract the weight values of every accepted live point as a list."""
    weighted_samples = open(file_weighted_samples)

    total_samples = 0
    for line in weighted_samples:
        total_samples += 1

    weighted_samples.seek(0)

    log_likelihoods = []

    for line in range(total_samples):
        weighted_samples.read(4)
        log_likelihoods.append(float(weighted_samples.read(24)))
        weighted_samples.readline()

    weighted_samples.close()

    return log_likelihoods


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
