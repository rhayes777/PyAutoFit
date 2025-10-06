from scipy.interpolate import CubicSpline
from .abstract import AbstractInterpolator


class SplineInterpolator(AbstractInterpolator):
    """
    Interpolate data with a piecewise cubic polynomial which is twice continuously differentiable
    """

    @staticmethod
    def _relationship(x, y):
        return CubicSpline(x, y)
