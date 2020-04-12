import logging
import math
import os
import pickle, pickle
import numpy as np
from dynesty import NestedSampler as StaticSampler
from dynesty.dynesty import DynamicNestedSampler
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

        **PyAutoFit** extends Dynesty by allowing runs to be terminated and resumed from that point. This is achieved
        by pickling the sampler instance during the model-fit after an input number of iterations.

        Attributes unique to **PyAutoFit** are described below, all remaining attributes are DyNesty parameters are
        described at the Dynesty API webpage:

        https://dynesty.readthedocs.io/en/latest/api.html#dynesty.dynamicsampler.stopping_function

        Parameters
        ----------
        paths : af.Paths
            A class that manages all paths, e.g. where the phase outputs are stored, the non-linear search chains,
            backups, etc.
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        terminate_at_acceptance_ratio : bool
            If *True*, the sampler will automatically terminate when the acceptance ratio falls behind an input
            threshold value (see *NestedSampler* for a full description of this feature).
        acceptance_ratio_threshold : float
            The acceptance ratio threshold below which sampling terminates if *terminate_at_acceptance_ratio* is
            *True* (see *NestedSampler* for a full description of this feature).

        Attributes
        ----------
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        iterations_per_update : int
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        """

        super().__init__(paths=paths, sigma=sigma)

        self.iterations_per_update = self.config("iterations_per_update", int)
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

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        """Copy this instance of the dynesty non-linear search with all associated attributes.

        This is used to set up the non-linear search on phase extensions."""
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.iterations_per_update = self.iterations_per_update
        copy.bound = self.bound
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

    def _simple_fit(self, model: AbstractPriorModel, fitness_function) -> Result:
        """
        Fit a model using Dynesty and a function that returns a likelihood from instances of that model.

        Dynesty is not called once, but instead called multiple times every iterations_per_update, such that the
        sampler instance can be pickled during the model-fit. This allows runs to be terminated and resumed.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        fitness_function
            A function that fits this model to the data, returning the likelihood of the fit.

        Returns
        -------
        A result object comprising the best-fit model instance, likelihood and an *Output* class that enables analysis
        of the full chains used by the fit.
        """
        dynesty_output = DynestyOutput(model=model, paths=self.paths)

        if os.path.exists("{}/{}.pickle".format(self.paths.backup_path, "dynesty")):

            with open(
                "{}/{}.pickle".format(self.paths.backup_path, "dynesty"), "rb"
            ) as f:
                dynesty_sampler = pickle.load(f)

        else:

            dynesty_sampler = self.sampler_fom_model_and_fitness(
                model=model, fitness_function=fitness_function
            )

        # These hacks are necessary to be able to pickle the sampler.

        dynesty_sampler.rstate = np.random
        pool = Pool(processes=1)
        dynesty_sampler.pool = pool
        dynesty_sampler.M = pool.map

        dynesty_finished = False

        while dynesty_finished is False:

            try:
                iterations_before_run = np.sum(dynesty_sampler.results.ncall)
            except AttributeError:
                iterations_before_run = 0

            dynesty_sampler.run_nested(maxcall=self.iterations_per_update)

            iterations_after_run = np.sum(dynesty_sampler.results.ncall)

            with open(
                "{}/{}.pickle".format(self.paths.backup_path, "dynesty"), "wb"
            ) as f:
                pickle.dump(dynesty_sampler, f)

            if iterations_before_run == iterations_after_run:
                dynesty_finished = True

        #        self.paths.backup()

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
        """Create this non-linear search's output class from the model and paths.

        This function is required by the aggregator, so it knows which output class to generate an instance of."""
        return DynestyOutput(model=model, paths=paths)

    def sampler_fom_model_and_fitness(self, model, fitness_function):
        return NotImplementedError()


class DynestyStatic(AbstractDynesty):
    def __init__(self, paths=None, sigma=3):
        """
        Class to setup and run a Dynesty non-linear search, using the static Dynesty nested sampler described at this
        webpage:

        https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.dynesty

        See the docstring for *AbstractDynesty* for complete documentation of Dynesty samplers.

        Attributes
        ----------
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        iterations_per_update : int
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        """

        super().__init__(paths=paths, sigma=sigma)

        self.n_live_points = self.config("n_live_points", int)

        logger.debug("Creating DynestyStatic NLO")

    @property
    def name(self):
        return "dynesty_static"

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        """Copy this instance of the dynesty non-linear search with all associated attributes.

        This is used to set up the non-linear search on phase extensions."""
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.n_live_points = self.n_live_points

        return copy

    def sampler_fom_model_and_fitness(self, model, fitness_function):
        """Get the static Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables."""

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


class DynestyDynamic(AbstractDynesty):
    def __init__(self, paths=None, sigma=3):
        """
        Class to setup and run a Dynesty non-linear search, using the dynamic Dynesty nested sampler described at this
        webpage:

        https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.dynamicsampler

        See the docstring for *AbstractDynesty* for complete documentation of Dynesty samplers.

        Attributes
        ----------
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        iterations_per_update : int
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        """

        super().__init__(paths=paths, sigma=sigma)

        logger.debug("Creating DynestyDynamic NLO")

    @property
    def name(self):
        return "dynesty_dynamic"

    def sampler_fom_model_and_fitness(self, model, fitness_function):
        """Get the dynamic Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables."""
        return DynamicNestedSampler(
            loglikelihood=fitness,
            prior_transform=prior,
            ndim=model.prior_count,
            logl_args=[model, fitness_function],
            ptform_args=[model],
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
        """The pickled instance of the *Dynesty* sampler, which provides access to all accept sample likelihoods,
        weights, etc. of the non-linear search.

        The sampler is described in the "Results" section at https://dynesty.readthedocs.io/en/latest/quickstart.html"""
        with open("{}/{}.pickle".format(self.paths.backup_path, "dynesty"), "rb") as f:
            return pickle.load(f)

    @property
    def results(self):
        """Convenience method to the pickled sample's results."""
        return self.sampler.results

    @property
    def pdf(self):
        """An interface to *GetDist* which can be used for analysing and visualizing the non-linear search chains.

        *GetDist* can only be used when chains are converged enough to provide a smooth PDF and this convergence is
        checked using the *pdf_converged* bool before *GetDist* is called.

        https://github.com/cmbant/getdist
        https://getdist.readthedocs.io/en/latest/

        For *Dynesty*, chains are passed to *GetDist* using the pickled sapler instance, which contains the physical
        model parameters of every accepted sample, their likelihoods and weights.
        """
        import getdist

        return getdist.mcsamples.MCSamples(
            samples=self.results.samples,
            weights=self.results.logwt,
            loglikes=self.results.logl,
        )

    @property
    def pdf_converged(self):
        """ To analyse and visualize chains using *GetDist*, the analysis must be sufficiently converged to produce
        smooth enough PDF for analysis. This property checks whether the non-linear search's chains are sufficiently
        converged for *GetDist* use.

        For *Dynesty*, during initial sampling one accepted live point typically has > 99% of the probabilty as its
        likelihood is significantly higher than all other points. Convergence is only achieved late in sampling when
        all live points have similar likelihood and sampling probabilities."""
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
    def number_live_points(self):
        """The number of live points used by the nested sampler."""
        return np.sum(self.results.nlive)

    @property
    def total_samples(self):
        """The total number of samples performed by the non-linear search.

        For Dynesty, this includes all accepted and rejected samples, and is loaded from the sampler pickle.
        """
        return np.sum(self.results.ncall)

    @property
    def total_accepted_samples(self):
        """The total number of accepted samples performed by the non-linear search.

        For Dynesty, this is loaded from the pickled sampler.
        """
        return self.results.niter

    @property
    def acceptance_ratio(self):
        """The ratio of accepted samples to total samples."""
        return self.total_accepted_samples / self.total_samples

    @property
    def maximum_log_likelihood(self):
        """The maximum log likelihood value of the non-linear search, corresponding to the best-fit model.

        For Dynesty, this is computed from the pickled sampler's list of all likelihood values."""
        return np.max(self.results.logl)

    @property
    def evidence(self):
        """The Bayesian evidence estimated by the nested sampling algorithm.

        For Dynesty, this is computed from the pickled sample's list of all evidence estimates."""
        return np.max(self.results.logz)

    @property
    def most_probable_vector(self):
        """ The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a list of values.

        If the chains are sufficiently converged this is estimated by passing the accepted samples to *GetDist*, else
        a crude estimate using the mean value of all accepted samples is used."""
        if self.pdf_converged:
            return self.pdf.getMeans()
        else:
            return list(np.mean(self.results.samples, axis=0))

    @property
    def most_likely_index(self):
        """The index of the accepted sample with the highest likelihood, e.g. that of best-fit / most_likely model."""
        return np.argmax(self.results.logl)

    @property
    def most_likely_vector(self):
        """ The best-fit model sampled by the non-linear search (corresponding to the maximum log-likelihood), returned
        as a list of values.

        The vector is read from the pickled sampler instance, by first locating the index corresponding to the highest
        likelihood accepted sample."""
        return self.results.samples[self.most_likely_index]

    def vector_at_sigma(self, sigma):
        """ The value of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as two lists of values corresponding to the lower and upper values parameter values.

        For example, if sigma is 1.0, the marginalized values of every parameter at 31.7% and 68.2% percentiles of each
        PDF is returned.

        This does not account for covariance between parameters. For example, if two parameters (x, y) are degenerate
        whereby x decreases as y gets larger to give the same PDF, this function will still return both at their
        upper values. Thus, caution is advised when using the function to reperform a model-fits.

        For *Dynesty*, this is estimated using *GetDist* if the chains have converged, by sampling the density
        function at an input PDF %. If not converged, a crude estimate using the range of values of the current
        physical live points is used.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        limit = math.erf(0.5 * sigma * math.sqrt(2))

        if self.pdf_converged:
            densities_1d = list(
                map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names)
            )

            return list(map(lambda p: p.getLimits(limit), densities_1d))
        else:

            parameters_min = list(
                np.min(self.results.samples[-self.number_live_points :], axis=0)
            )
            parameters_max = list(
                np.max(self.results.samples[-self.number_live_points :], axis=0)
            )

            return [
                (parameters_min[index], parameters_max[index])
                for index in range(len(parameters_min))
            ]

    def vector_from_sample_index(self, sample_index):
        """The model parameters of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return self.results.samples[sample_index]

    def weight_from_sample_index(self, sample_index):
        """The weight of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return self.results.logwt[sample_index]

    def likelihood_from_sample_index(self, sample_index):
        """The likelihood of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return self.results.logl[sample_index]
