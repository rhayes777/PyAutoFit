import os
import shutil
from os import path
from typing import Optional
from sqlalchemy.orm import Session

from autoconf import conf
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear import abstract_search
from autofit.non_linear import result as res
from autofit.non_linear.log import logger
from autofit.non_linear.nest import abstract_nest
from autofit.non_linear.samples import NestSamples, Sample


class MultiNest(abstract_nest.AbstractNest):
    __identifier_fields__ = (
        "n_live_points",
        "sampling_efficiency",
        "const_efficiency_mode",
        "importance_nested_sampling",
        "max_modes",
        "mode_tolerance",
        "seed",
    )

    def __init__(
            self,
            name=None,
            path_prefix=None,
            unique_tag: Optional[str] = None,
            prior_passer=None,
            session : Optional[Session] = None,
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
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        prior_passer
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            prior_passer=prior_passer,
            session=session,
            **kwargs
        )

        logger.debug("Creating MultiNest Search")

    class Fitness(abstract_nest.AbstractNest.Fitness):

        def __init__(
                self,
                paths,
                model,
                analysis,
                samples_from_model,
                stagger_resampling_likelihood,
                log_likelihood_cap=None
        ):

            super().__init__(
                model=model,
                analysis=analysis,
                samples_from_model=samples_from_model,
                stagger_resampling_likelihood=stagger_resampling_likelihood,
                log_likelihood_cap=log_likelihood_cap,
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
            **self.config_dict_search
        )
        self.copy_from_sym()

    def samples_from(self, model: AbstractPriorModel):
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

        return MultiNestSamples(
            model=model,
            number_live_points=self.config_dict_search["n_live_points"],
            file_summary=path.join(self.paths.samples_path, "multinestsummary.txt"),
            file_weighted_samples=path.join(self.paths.samples_path, "multinest.txt"),
            file_resume=path.join(self.paths.samples_path, "multinestresume.dat"),
            unconverged_sample_size=1,
            time=self.timer.time
        )

    def copy_from_sym(self):
        """
        Copy files from the sym-linked search folder to the samples folder.
        """

        src_files = os.listdir(self.paths.path)
        for file_name in src_files:
            full_file_name = path.join(self.paths.path, file_name)
            if path.isfile(full_file_name):
                shutil.copy(full_file_name, self.paths.samples_path)


class MultiNestSamples(NestSamples):

    def __init__(
            self,
            model: AbstractPriorModel,
            number_live_points : int,
            file_summary : str,
            file_weighted_samples : str,
            file_resume : str,
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

        super().__init__(
            model=model,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
        )

        self._samples = None
        self._number_live_points = number_live_points

    @property
    def samples(self):
        """
        Create a `Samples` object from this non-linear search's output files on the hard-disk and model.

        For Emcee, all quantities are extracted via the hdf5 backend of results.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the `NonLinearSearch` chains,
            etc.
        """

        if self._samples is not None:
            return self._samples

        parameters = parameters_from_file_weighted_samples(
            file_weighted_samples=self.file_weighted_samples,
            prior_count=self.model.prior_count,
        )

        log_prior_list = [
            sum(self.model.log_prior_list_from_vector(vector=vector)) for vector in parameters
        ]

        log_likelihood_list = log_likelihood_list_from_file_weighted_samples(
            file_weighted_samples=self.file_weighted_samples
        )

        weight_list = weight_list_from_file_weighted_samples(
            file_weighted_samples=self.file_weighted_samples
        )

        self._samples = Sample.from_lists(
            model=self.model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return self._samples

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
