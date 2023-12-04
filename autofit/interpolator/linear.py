from scipy.stats import stats
from .abstract import AbstractInterpolator


class LinearInterpolator(AbstractInterpolator):
    """
    Assume all attributes have a linear relationship with time
    """

    @staticmethod
    def _interpolate(x, y, value):
        slope, intercept, r, p, std_err = stats.linregress(x, y)
        return slope * value + intercept
