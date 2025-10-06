from scipy.stats import linregress
from .abstract import AbstractInterpolator
from autofit.interpolator.linear_relationship import LinearRelationship


class LinearInterpolator(AbstractInterpolator):
    """
    Assume all attributes have a linear relationship with time
    """

    @staticmethod
    def _relationship(x, y):
        slope, intercept, r, p, std_err = linregress(x, y)

        return LinearRelationship(slope, intercept)
