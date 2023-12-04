from scipy.interpolate import CubicSpline
from .abstract import AbstractInterpolator


class SplineInterpolator(AbstractInterpolator):
    """
    Interpolate data with a piecewise cubic polynomial which is twice continuously differentiable
    """

    @staticmethod
    def _interpolate(x, y, value):
        f = CubicSpline(x, y)
        return f(value)
