from scipy.stats import linregress
from .abstract import AbstractInterpolator


class LinearInterpolator(AbstractInterpolator):
    """
    Assume all attributes have a linear relationship with time
    """

    @staticmethod
    def _interpolate(x, y, value):
        slope, intercept, r, p, std_err = linregress(x, y)
        return slope * value + intercept
