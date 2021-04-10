import numpy as np


class AutoCorrelationsSettings:

    def __init__(
            self,
            check_for_convergence=None,
            check_size=None,
            required_length=None,
            change_threshold=None,
    ):
        """
        Class for performing and customizing AutoCorrelation calculations, which are used:

         - By the `Samples` object during a model-fit to determine is an ensemble MCMC sampler should terminate.
         - After a model-fit is finished to investigate whether the resutls converged.

        Parameters
        ----------
        check_for_convergence : bool
            Whether the auto-correlation lengths of the Emcee samples are checked to determine the stopping criteria.
            If `True`, this option may terminate the Emcee run before the input number of steps, nsteps, has
            been performed. If `False` nstep samples will be taken.
        check_size : int
            The length of the samples used to check the auto-correlation lengths (from the latest sample backwards).
            For convergence, the auto-correlations must not change over a certain range of samples. A longer check-size
            thus requires more samples meet the auto-correlation threshold, taking longer to terminate sampling.
            However, shorter chains risk stopping sampling early due to noise.
        required_length : int
            The length an auto_correlation chain must be for it to be used to evaluate whether its change threshold is
            sufficiently small to terminate sampling early.
        change_threshold : float
            The threshold value by which if the change in auto_correlations is below sampling will be terminated early.
        """
        self.check_for_convergence = check_for_convergence
        self.check_size = check_size
        self.required_length = required_length
        self.change_threshold = change_threshold

    def update_via_config(self, config):
        config_dict = config._dict

        self.check_for_convergence = (
            config_dict["check_for_convergence"]
            if self.check_for_convergence is None
            else self.check_for_convergence
        )
        self.check_size = (
            config_dict["check_size"]
            if self.check_size is None
            else self.check_size
        )
        self.required_length = (
            config_dict["required_length"]
            if self.required_length is None
            else self.required_length
        )
        self.change_threshold = (
            config_dict["change_threshold"]
            if self.change_threshold is None
            else self.change_threshold
        )


class AutoCorrelations(AutoCorrelationsSettings):

    def __init__(
            self,
            check_size,
            required_length,
            change_threshold,
            times,
            previous_times,

    ):
        """
        Class for performing and customizing AutoCorrelation calculations, which are used:

         - By the `Samples` object during a model-fit to determine is an ensemble MCMC sampler should terminate.
         - After a model-fit is finished to investigate whether the resutls converged.

        Parameters
        ----------
        check_for_convergence : bool
            Whether the auto-correlation lengths of the Emcee samples are checked to determine the stopping criteria.
            If `True`, this option may terminate the Emcee run before the input number of steps, nsteps, has
            been performed. If `False` nstep samples will be taken.
        check_size : int
            The length of the samples used to check the auto-correlation lengths (from the latest sample backwards).
            For convergence, the auto-correlations must not change over a certain range of samples. A longer check-size
            thus requires more samples meet the auto-correlation threshold, taking longer to terminate sampling.
            However, shorter chains risk stopping sampling early due to noise.
        required_length : int
            The length an auto_correlation chain must be for it to be used to evaluate whether its change threshold is
            sufficiently small to terminate sampling early.
        change_threshold : float
            The threshold value by which if the change in auto_correlations is below sampling will be terminated early.
        """

        super().__init__(
            check_size=check_size,
            required_length=required_length,
            change_threshold=change_threshold
        )

        self.times = times
        self.previous_times = previous_times

    @property
    def relative_times(self) -> [float]:
        return np.abs(self.previous_times - self.times) / self.times

    def check_if_converged(self, total_samples):

        """
        Whether the emcee samples have converged on a solution or if they are still in a burn-in period, based on the
        auto correlation times of parameters.
        """
        converged = np.all(
            self.times * self.required_length
            < total_samples
        )
        if converged:
            try:
                converged &= np.all(self.relative_times < self.change_threshold)
            except IndexError:
                return False
        return converged
