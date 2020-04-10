import logging
import math
import os
import pickle
import numpy as np
from dynesty import NestedSampler
from multiprocessing.pool import Pool

from autofit import conf, exc
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.optimize.non_linear.non_linear import NonLinearOptimizer
from autofit.optimize.non_linear.non_linear import Result
from autofit.optimize.non_linear.output import NestedSamplingOutput
from autofit.optimize.non_linear.paths import Paths

logger = logging.getLogger(__name__)

# Pickling does not work if its in the scope of the Dynesty class

def prior(cube, model):

    # YOU MAY REFACTOR THIS LINE

    return model.vector_from_unit_vector(unit_vector=cube)


def fitness(cube, model, fitness_function):
    return fitness_function(
        model.instance_from_vector(
            cube
        )
    )


class Dynesty(NonLinearOptimizer):
    def __init__(self, paths=None, sigma=3):
        """
        Class to setup and run a Dynesty lens and output the MultiNest nlo.

        This interfaces with an input model_mapper, which is used for setting up the \
        individual model instances that are passed to each iteration of MultiNest.
        """
        if paths is None:
            paths = Paths()

        super().__init__(paths)

        self.sigma = sigma

        self.terminate_at_acceptance_ratio = self.config(
             "terminate_at_acceptance_ratio", bool
        )
        self.acceptance_ratio_threshold = self.config(
            "acceptance_ratio_threshold", float
        )

        logger.debug("Creating Dynesty NLO")

    @property
    def name(self):
        return "dynesty"

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        return copy

    class Fitness(NonLinearOptimizer.Fitness):
        def __init__(self, paths, analysis, dynesty_output, terminate_at_acceptance_ratio,
                     acceptance_ratio_threshold):
            super().__init__(paths, analysis, dynesty_output.output_results)
            self.accepted_samples = 0
            self.dynesty_output = dynesty_output

            self.model_results_output_interval = conf.instance.general.get(
                "output", "model_results_output_interval", int
            )
            self.stagger_resampling_likelihood = conf.instance.non_linear.get(
                "MultiNest", "stagger_resampling_likelihood", bool
            )
            self.stagger_resampling_value = conf.instance.non_linear.get(
                "MultiNest", "stagger_resampling_value", float
            )
            self.resampling_likelihood = conf.instance.non_linear.get(
                "MultiNest", "null_log_evidence", float
            )
            self.terminate_at_acceptance_ratio = terminate_at_acceptance_ratio
            self.acceptance_ratio_threshold = acceptance_ratio_threshold

            self.terminate_has_begun = False
            self.stagger_accepted_samples = 0

        def __call__(self, instance):

            if self.terminate_at_acceptance_ratio:
                if os.path.isfile(self.paths.file_summary):
                    try:
                        if (
                                self.dynesty_output.acceptance_ratio < self.acceptance_ratio_threshold) or self.terminate_has_begun:
                            self.terminate_has_begun = True
                            return self.max_likelihood
                    except ValueError:
                        pass

            try:
                likelihood = self.fit_instance(instance)
            except exc.FitException:

                if not self.stagger_resampling_likelihood:
                    likelihood = -np.inf
                else:

                    if self.stagger_accepted_samples < 10:

                        self.stagger_accepted_samples += 1
                        self.resampling_likelihood += self.stagger_resampling_value
                        likelihood = self.resampling_likelihood

                    else:

                        likelihood = -1.0 * np.abs(self.resampling_likelihood) * 10.0

            return likelihood

    def _simple_fit(self, model: AbstractPriorModel, fitness_function) -> Result:
        """
        Fit a model using MultiNest and some function that
        scores instances of that model.

        Parameters
        ----------
        model
            The model which is used to generate instances for different
            points in parameter space
        fitness_function
            A function that gives a score to the model, with the highest (least
            negative) number corresponding to the best fit.

        Returns
        -------
        A result object comprising a fitness score, model instance and model.
        """
        dynesty_output = DynestyOutput(model, self.paths)

        if os.path.exists("{}/{}.pickle".format(self.paths.sym_path, "nls")):
            with open("{}/{}.pickle".format(self.paths.sym_path, "nls"), "rb") as f:
                dynesty_sampler = pickle.load(f)

        else:

            dynesty_sampler = NestedSampler(loglikelihood=fitness, prior_transform=prior, ndim=model.prior_count,
                                            logl_args=[model, fitness_function], ptform_args=[model])

        dynesty_sampler.rstate = np.random
        pool = Pool(processes=1)
        dynesty_sampler.pool = pool
        dynesty_sampler.M = pool.map

        dynesty_sampler.run_nested(maxcall=2000)

        with open("{}/{}.pickle".format(self.paths.sym_path, "nls"), "wb") as f:
            pickle.dump(dynesty_sampler, f)

        print(dynesty_sampler.results.summary())

        self.paths.backup()

        instance = dynesty_output.most_likely_instance
        dynesty_output.output_results(
            during_analysis=False
        )
        return Result(
            instance=instance,
            likelihood=dynesty_output.maximum_log_likelihood,
            output=dynesty_output,
            previous_model=model,
            gaussian_tuples=dynesty_output.gaussian_priors_at_sigma(self.sigma),
        )

    def _fit(self, analysis, model):
        dynesty_output = DynestyOutput(model, self.paths)

        dynesty_output.save_model_info()

        if not os.path.exists(self.paths.has_completed_path):
            fitness_function = Dynesty.Fitness(
                self.paths,
                analysis,
                dynesty_output,
                self.terminate_at_acceptance_ratio,
                self.acceptance_ratio_threshold
            )

            logger.info("Running MultiNest...")
            self._simple_fit(
                model,
                fitness_function.__call__
            )
            logger.info("MultiNest complete")


            self.paths.backup()
            open(self.paths.has_completed_path, "w+").close()
        else:
            logger.warning(
                f"{self.paths.phase_name} has run previously - skipping"
            )

        instance = dynesty_output.most_likely_instance
        analysis.visualize(instance=instance, during_analysis=False)
        dynesty_output.output_results(during_analysis=False)
        dynesty_output.output_pdf_plots()
        result = Result(
            instance=instance,
            likelihood=dynesty_output.maximum_log_likelihood,
            output=dynesty_output,
            previous_model=model,
            gaussian_tuples=dynesty_output.gaussian_priors_at_sigma(self.sigma),
        )
        self.paths.backup_zip_remove()
        return result

    def output_from_model(self, model, paths):
        return DynestyOutput(model=model, paths=paths)


class DynestyOutput(NestedSamplingOutput):
    @property
    def pdf(self):
        import getdist

        try:
            return getdist.mcsamples.loadMCSamples(
                self.paths.backup_path + "/dynesty"
            )
        except IOError or OSError or ValueError or IndexError:
            raise Exception

    @property
    def pdf_converged(self):
        try:
            densities_1d = list(
                map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names)
            )

            if densities_1d == []:
                return False

            return True
        except Exception:
            return False

    @property
    def most_probable_vector(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        dynesty lens.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.

        """
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=self.model.prior_count, offset=0
            )
        except FileNotFoundError:
            return self.most_likely_vector

    @property
    def most_likely_vector(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        dynesty lens.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.
        """
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=self.model.prior_count, offset=56
            )
        except FileNotFoundError:
            most_likey_index = np.argmax([point[-1] for point in self.phys_live_points])
            return self.phys_live_points[most_likey_index][0:-1]

    @property
    def maximum_log_likelihood(self):
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=2, offset=112
            )[1]
        except FileNotFoundError:
            return max([point[-1] for point in self.phys_live_points])

    @property
    def evidence(self):
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=2, offset=112
            )[0]
        except FileNotFoundError:
            return None

    @property
    def phys_live_points(self):

        phys_live = open(self.paths.file_phys_live)

        live_points = 0
        for line in phys_live:
            live_points += 1

        phys_live.seek(0)

        phys_live_points = []

        for line in range(live_points):
            vector = []
            for param in range(self.model.prior_count + 1):
                vector.append(float(phys_live.read(28)))
            phys_live.readline()
            phys_live_points.append(vector)

        phys_live.close()

        return phys_live_points

    def phys_live_points_of_param(self, param_index):
        return [point[param_index] for point in self.phys_live_points]

    def read_list_of_results_from_summary_file(self, number_entries, offset):

        summary = open(self.paths.file_summary)
        summary.read(2 + offset * self.model.prior_count)
        vector = []
        for param in range(number_entries):
            vector.append(float(summary.read(28)))

        summary.close()

        return vector

    def vector_at_sigma(self, sigma):
        limit = math.erf(0.5 * sigma * math.sqrt(2))

        if self.pdf_converged:
            densities_1d = list(
                map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names)
            )

            return list(map(lambda p: p.getLimits(limit), densities_1d))
        else:

            parameters_min = [
                min(self.phys_live_points_of_param(param_index=param_index))
                for param_index in range(self.model.prior_count)
            ]
            parameters_max = [
                max(self.phys_live_points_of_param(param_index=param_index))
                for param_index in range(self.model.prior_count)
            ]

            return [
                (parameters_min[index], parameters_max[index])
                for index in range(len(parameters_min))
            ]

    @property
    def total_samples(self):
        resume = open(self.paths.file_resume)

        resume.seek(1)
        resume.read(19)
        return int(resume.read(8))

    @property
    def accepted_samples(self):

        resume = open(self.paths.file_resume)

        resume.seek(1)
        resume.read(8)
        return int(resume.read(10))

    @property
    def acceptance_ratio(self):
        return self.accepted_samples / self.total_samples

    def vector_from_sample_index(self, sample_index):
        """From a sample return the model parameters.

        Parameters
        -----------
        sample_index : int
            The sample index of the weighted sample to return.
        """
        return list(self.pdf.samples[sample_index])

    def weight_from_sample_index(self, sample_index):
        """From a sample return the sample weight.

        Parameters
        -----------
        sample_index : int
            The sample index of the weighted sample to return.
        """
        return self.pdf.weights[sample_index]

    def likelihood_from_sample_index(self, sample_index):
        """From a sample return the likelihood.

        NOTE: GetDist reads the log likelihood from the weighted_sample.txt file (column 2), which are defined as \
        -2.0*likelihood. This routine converts these back to likelihood.

        Parameters
        -----------
        sample_index : int
            The sample index of the weighted sample to return.
        """
        return -0.5 * self.pdf.loglikes[sample_index]
