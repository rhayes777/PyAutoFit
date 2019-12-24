""" Implements a NUTS Sampler for emcee
    http://dan.iel.fm/emcee/
"""
import numpy as np
from .nuts import nuts6
from .helpers import NutsSampler_fn_wrapper
from emcee.ptsampler import PTSampler


__all__ = ["NUTSSampler"]


class NUTSSampler(PTSampler):
    """ A sampler object mirroring emcee.sampler object definition"""

    def __init__(self, dim, log_prob_fn, gradient_fn=None, *args, **kwargs):

        super(NUTSSampler, self).__init__()

        self.dim = dim
        self.f = NutsSampler_fn_wrapper(log_prob_fn, gradient_fn, *args, **kwargs)
        self.log_prob_fn = self.f.lnp_func
        self.gradient_fn = self.f.gradlnp_func
        self.reset()

    @property
    def random_state(self):
        """
        The state of the internal random number generator. In practice, it's
        the result of calling ``get_state()`` on a
        ``numpy.random.mtrand.RandomState`` object. You can try to set this
        property but be warned that if you do this and it fails, it will do
        so silently.
        """
        pass

    @random_state.setter  # NOQA
    def random_state(self, state):
        """
        Try to set the state of the random number generator but fail silently
        if it doesn't work. Don't say I didn't warn you...

        """
        pass

    @property
    def flatlnprobability(self):
        """
        A shortcut to return the equivalent of ``lnprobability`` but aligned
        to ``flatchain`` rather than ``chain``.

        """
        return self.lnprobability.flatten()

    def get_lnprob(self, p):
        """Return the log-probability at the given position."""
        return self.log_prob_fn(p)

    def get_gradlnprob(self, p, dx=1e-3, order=1):
        """Return the log-probability at the given position."""
        return self.gradient_fn(p)

    def reset(self):
        """
        Clear ``chain``, ``lnprobability`` and the bookkeeping parameters.

        """
        self._lnprob = []
        self._chain = []
        self._epsilon = 0.0

    @property
    def iterations(self):
        return len(self._lnprob)

    def clear_chain(self):
        """An alias for :func:`reset` kept for backwards compatibility."""
        return self.reset()

    def _sample_fn(self, p, dx=1e-3, order=1):
        """ proxy function for nuts6 """
        lnprob = self.log_prob_fn(p)
        gradlnp = self.gradient_fn(p)
        return (lnprob, gradlnp)

    def sample(self, initial_state, steps, steps_burn_in, delta=0.6, **kwargs):
        """ Runs NUTS6 """
        samples, lnprob, epsilon = nuts6(
            self._sample_fn, steps, steps_burn_in, initial_state, delta
        )
        self._chain = samples
        self._lnprob = lnprob
        self._epsilon = epsilon

        return samples

    def run_mcmc(self, initial_state, steps, steps_burn_in, delta=0.6, **kwargs):
        """
        Iterate :func:`sample` for ``N`` iterations and return the result.

        :param initial_state:
            The initial position vector.

        :param steps:
            The number of steps to run.

        :param steps_burn_in:
            The number of steps to run during the burning period.

        :param delta: (optional, default=0.6)
            Initial step size.

        :param kwargs: (optional)
            Other parameters that are directly passed to :func:`sample`.

        """

        print("Running HMC with dual averaging and trajectory length %0.2f..." % delta)
        return self.sample(initial_state, steps, steps_burn_in, delta, **kwargs)


class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    are also included.

    """

    def __init__(self, f, args):
        self.f = f
        self.args = args

    def __call__(self, x):
        try:
            return self.f(x, *self.args)
        except:
            import traceback

            print("NUTS: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  exception:")
            traceback.print_exc()
            raise
