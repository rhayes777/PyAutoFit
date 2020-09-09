from functools import reduce
from operator import mul
from typing import (
    Iterable, Tuple, TypeVar, Dict
)

import numpy as np
from scipy import special

from autofit.mapper.variable import Variable


class FlattenArrays(dict):
    """
    >>> shapes = FlattenArrays(a=(1, 2), b=(2, 3))
    >>> shapes
    FlattenArrays(a=(1, 2), b=(2, 3))
    >>> shapes.flatten(
        a = np.arange(2).reshape(1, 2),
        b = np.arange(6).reshape(2, 3)**2)
    array([ 0,  1,  0,  1,  4,  9, 16, 25])
    >>> shapes.unflatten(
        [ 0,  1,  0,  1,  4,  9, 16, 25])
    {'a': array([[0, 1]]), 'b': array([[ 0,  1,  4],
        [ 9, 16, 25]])}
    """

    def __init__(self, dict_: Dict[Variable, Tuple[int, ...]]):
        super().__init__()

        self.update(dict_)
        self.splits = np.cumsum([
            np.prod(s) for s in self.values()], dtype=int)
        self.inds = [
            np.arange(i0, i1, dtype=int) for i0, i1 in
            zip(np.r_[0, self.splits[:-1]], self.splits)]

    @classmethod
    def from_arrays(cls, **arrays: Dict[str, np.ndarray]) -> "FlattenArrays":
        return cls(**{k: np.shape(arr) for k, arr in arrays.items()})

    def flatten(self, arrays_dict: Dict[Variable, np.ndarray]) -> np.ndarray:
        assert all(np.shape(arrays_dict[k]) == shape
                   for k, shape in self.items())
        return np.concatenate([
            np.ravel(arrays_dict[k]) for k in self.keys()])

    def unflatten(self, arr: np.ndarray, ndim=None) -> Dict[str, np.ndarray]:
        arr = np.asanyarray(arr)
        if ndim is None:
            ndim = arr.ndim
        arrays = [arr[np.ix_(*(ind for _ in range(ndim)))] for ind in self.inds]
        arr_shapes = [arr.shape[ndim:] for arr in arrays]
        return {
            k: arr.reshape(shape * ndim + arr_shape)
            if shape or arr_shape else arr.item()
            for (k, shape), arr_shape, arr in
            zip(self.items(), arr_shapes, arrays)}

    def __repr__(self):
        shapes = ", ".join(map("{0[0]}={0[1]}".format, self.items()))
        return f"{type(self).__name__}({shapes})"

    @property
    def size(self):
        return self.splits[-1]


def add_arrays(*arrays: np.ndarray) -> np.ndarray:
    """Sums over broadcasting multidimensional arrays
    whilst preserving the total sum

    a = np.arange(10).reshape(1, 2, 1, 5)
    b = np.arange(8).reshape(2, 2, 2, 1)

    >>> add_arrays(a, b).sum()
    73.0
    >>> add_arrays(a, b).shape
    (2, 2, 2, 5)
    >>> a.sum() + b.sum()
    73
    """
    b = np.broadcast(*arrays)
    return sum(a * np.size(a) / b.size for a in arrays)


_M = TypeVar('M')


def prod(iterable: Iterable[_M], *arg: Tuple[_M]) -> _M:
    """calculates the product of the passed iterable,
    much like sum, if a second argument is passed,
    this is the inital value of the calculation

    Examples
    --------
    >>> prod(range(1, 3))
    2

    >>> prod(range(1, 3), 2.)
    4.
    """
    return reduce(mul, iterable, *arg)


def propagate_uncertainty(
        cov: np.ndarray, jac: np.ndarray) -> np.ndarray:
    """Propagates the uncertainty of a covariance matrix given the
    passed Jacobian

    If the variable arrays are multidimensional then will output in
    the shape of the arrays

    see https://en.wikipedia.org/wiki/Propagation_of_uncertainty
    """
    cov = np.asanyarray(cov)

    var_ndim = cov.ndim // 2
    det_ndim = jac.ndim - var_ndim
    det_shape, var_shape = jac.shape[:det_ndim], jac.shape[det_ndim:]
    assert var_shape == cov.shape[:var_ndim] == cov.shape[var_ndim:]

    var_size = np.prod(var_shape, dtype=int)
    det_size = np.prod(det_shape, dtype=int)

    cov2d = cov.reshape((var_size, var_size))
    jac2d = jac.reshape((det_size, var_size))

    det_cov2d = np.linalg.multi_dot((
        jac2d, cov2d, jac2d.T))
    det_cov = det_cov2d.reshape(det_shape + det_shape)
    return det_cov


def psilog(x: np.ndarray) -> np.ndarray:
    """
    psi(x) - log(x)
    needed when calculating E[ln[x]] when x is a Gamma variable
    """
    return special.digamma(x) - np.log(x)


def grad_psilog(x: np.ndarray) -> np.ndarray:
    """d_x (psi(x) - log(x)) = psi^1(x) - 1/x

    needed when calculating the inverse of psilog(x)
    by using Newton-Raphson

    see:
    invpsilog(c)
    """
    return special.polygamma(1, x) - 1 / x


def invpsilog(c: np.ndarray) -> np.ndarray:
    """
    Solves the equation

    psi(x) - log(x) = c

    where psi is the digamma function. c must be negative.
    The function calculates an approximate inverse which it uses as
    a starting point to 4 iterations of the Newton-Raphson algorithm.
    """
    c = np.asanyarray(c)

    if not np.all(c < 0):
        raise ValueError("values passed must be negative")

    # approximate starting guess
    # -1/x < psilog(x) < -1/(2x)
    A, beta, gamma = 0.38648347, 0.89486989, 0.78578843
    x0 = -(1 - 0.5 * (1 + A * (-c) ** beta) ** -gamma) / c

    # do 4 iterations of Newton Raphson to refine estimate
    for i in range(4):
        f0 = psilog(x0) - c
        x0 = x0 - f0 / grad_psilog(x0)

    return x0
