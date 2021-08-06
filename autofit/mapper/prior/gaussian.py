import math

from scipy.special import erfcinv

from autofit.messages.normal import NormalMessage


class GaussianPrior(NormalMessage):
    """A prior with a gaussian distribution"""



    def __call__(self, x):
        return self.logpdf(x)
