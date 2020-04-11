import logging
import math
import os
import pickle
import numpy as np
from dynesty import NestedSampler as StaticSampler
from dynesty.dynamicsampler import DynamicSampler
from multiprocessing.pool import Pool

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.optimize.non_linear.nested_sampling.nested_sampler import (
    NestedSampler,
    NestedSamplerOutput,
)
from autofit.optimize.non_linear.non_linear import Result

logger = logging.getLogger(__name__)

# Pickling does not work if its in the scope of the Dynesty class


def prior(cube, model):

    # YOU MAY REFACTOR THIS LINE

    return model.vector_from_unit_vector(unit_vector=cube)


def fitness(cube, model, fitness_function):
    return fitness_function(model.instance_from_vector(cube))


class AbstractDynesty(NestedSampler):

    def __init__(self, paths=None, sigma=3):
        """
        Class to setup and run a Dynesty non-linear search.

        For a full description of Dynesty, checkout its GitHub and readthedocs webpages:

        https://github.com/joshspeagle/dynesty

        https://dynesty.readthedocs.io/en/latest/index.html

        Attributes
        ----------
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        iterations_per_update : int
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).

        All remaining attributes are DyNesty parameters and described at the Dynesty API webpage:

        https://dynesty.readthedocs.io/en/latest/api.html#dynesty.dynamicsampler.stopping_function


        """

        super().__init__(paths=paths, sigma=sigma)

        self.bound = self.config("bound", str)

        logger.debug("Creating DynestyStatic NLO")

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )

        copy.bound = self.bound

        return copy

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

        if os.path.exists("{}/{}.pickle".format(self.paths.backup_path, "dynesty")):

            with open(
                "{}/{}.pickle".format(self.paths.backup_path, "dynesty"), "rb"
            ) as f:
                dynesty_sampler = pickle.load(f)

        else:

            dynesty_sampler = self.sampler_fom_model_and_fitness(model=model, fitness_function=fitness_function)

        dynesty_sampler.rstate = np.random
        pool = Pool(processes=1)
        dynesty_sampler.pool = pool
        dynesty_sampler.M = pool.map

        dynesty_finished = False

        while dynesty_finished is False:

            try:
                iterations_before_run = len(dynesty_sampler.results["ncall"])
            except AttributeError:
                iterations_before_run = 0

            dynesty_sampler.run_nested(maxcall=self.iterations_per_update)

            iterations_after_run = len(dynesty_sampler.results["ncall"])

            with open(
                "{}/{}.pickle".format(self.paths.backup_path, "dynesty"), "wb+"
            ) as f:
                pickle.dump(dynesty_sampler, f)

            if iterations_before_run == iterations_after_run:
                dynesty_finished = True

        self.paths.backup()

        print(dynesty_output.pdf.getMeans())

        instance = dynesty_output.most_likely_instance
        dynesty_output.output_results(during_analysis=False)
        return Result(
            instance=instance,
            likelihood=dynesty_output.maximum_log_likelihood,
            output=dynesty_output,
            previous_model=model,
            gaussian_tuples=dynesty_output.gaussian_priors_at_sigma(self.sigma),
        )

    def output_from_model(self, model, paths):
        return DynestyOutput(model=model, paths=paths)

    def sampler_fom_model_and_fitness(self, model, fitness_function):
        return NotImplementedError()

class DynestyStatic(AbstractDynesty):

    def __init__(self, paths=None, sigma=3):
        """
        Class to setup and run a Dynesty non-linear search.

        For a full description of Dynesty, checkout its GitHub and readthedocs webpages:

        https://github.com/joshspeagle/dynesty

        https://dynesty.readthedocs.io/en/latest/index.html

        Attributes
        ----------
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        iterations_per_update : int
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).

        All remaining attributes are DyNesty parameters and described at the Dynesty API webpage:

        https://dynesty.readthedocs.io/en/latest/api.html#dynesty.dynamicsampler.stopping_function


        """

        super().__init__(paths=paths, sigma=sigma)

        self.iterations_per_update = self.config("iterations_per_update", int)
        self.n_live_points = self.config("n_live_points", int)
        self.bound = self.config("bound", str)
        self.sample = self.config("sample", str)
        self.bootstrap = self.config("bootstrap", int)
        self.enlarge = self.config("enlarge", float)

        self.update_interval = self.config("update_interval", float)

        if self.update_interval < 0.0:
            self.update_interval = None

        if self.enlarge < 0.0:
            if self.bootstrap == 0.0:
                self.enlarge = 1.0
            else:
                self.enlarge = 1.25

        self.vol_dec = self.config("vol_dec", float)
        self.vol_check = self.config("vol_check", float)
        self.walks = self.config("walks", int)
        self.facc = self.config("facc", float)
        self.slices = self.config("slices", int)
        self.fmove = self.config("fmove", float)
        self.max_move = self.config("max_move", int)

        logger.debug("Creating DynestyStatic NLO")

    @property
    def name(self):
        return "dynesty_static"

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.iterations_per_update = self.iterations_per_update
        copy.n_live_points = self.n_live_points
        copy.sample = self.sample
        copy.update_interval = self.update_interval
        copy.bootstrap = self.bootstrap
        copy.enlarge = self.enlarge
        copy.vol_dec = self.vol_dec
        copy.vol_check = self.vol_check
        copy.walks = self.walks
        copy.facc = self.facc
        copy.slices = self.slices
        copy.fmove = self.fmove
        copy.max_move = self.max_move

        return copy

    def sampler_fom_model_and_fitness(self, model, fitness_function):

        return StaticSampler(
            loglikelihood=fitness,
            prior_transform=prior,
            ndim=model.prior_count,
            logl_args=[model, fitness_function],
            ptform_args=[model],
            nlive=self.n_live_points,
            bound=self.bound,
            sample=self.sample,
            update_interval=self.update_interval,
            bootstrap=self.bootstrap,
            enlarge=self.enlarge,
            vol_dec=self.vol_dec,
            vol_check=self.vol_check,
            walks=self.walks,
            facc=self.facc,
            slices=self.slices,
            fmove=self.fmove,
            max_move=self.max_move,
        )

class DynestyOutput(NestedSamplerOutput):
    @property
    def sampler(self):
        with open("{}/{}.pickle".format(self.paths.backup_path, "dynesty"), "rb") as f:
            return pickle.load(f)

    @property
    def results(self):
        return self.sampler.results

    @property
    def pdf(self):
        import getdist

        print(self.results.samples)
        print(self.results.logwt)

        return getdist.mcsamples.MCSamples(
            samples=self.results.samples,
            weights=self.results.logwt,
            loglikes=self.results.logl,
        )

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
