from operator import mul
from functools import reduce
from collections import namedtuple
from typing import (
    Iterable, Tuple, List, Optional, Any, TypeVar, Callable,
    Dict
)

import numpy as np
from scipy import special


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

    def __init__(self, *args, **kwargs: Dict[str, Tuple[int, ...]]):
        super().__init__(*args, **kwargs)
        self.splits = np.cumsum([
            np.prod(s) for s in self.values()], dtype=int)
        self.inds = [
            np.arange(i0, i1, dtype=int) for i0, i1 in
            zip(np.r_[0, self.splits[:-1]], self.splits)]

    @classmethod
    def from_arrays(cls, **arrays: Dict[str, np.ndarray]) -> "FlattenArrays":
        return cls(**{k: np.shape(arr) for k, arr in arrays.items()})

    def flatten(self, **arrays: Dict[str, np.ndarray]) -> np.ndarray:
        assert all(np.shape(arrays[k]) == shape
                   for k, shape in self.items())
        return np.concatenate([
            np.ravel(arrays[k]) for k in self.keys()])

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

    def _immutable(self, *args, **kws):
        raise TypeError('object is immutable')

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
    pop = _immutable
    popitem = _immutable


def line_count(filepath):
    with open(filepath) as f:
        return sum(1 for l in f if l.strip())


def jupyter_word_count(filepath, as_version=2):
    import io
    from nbformat import read
    with io.open(filepath, 'r', encoding='utf-8') as f:
        nb = read(f, as_version)

    word_count = 0
    for cell in nb.worksheets[0].cells:
        if cell.cell_type == "markdown":
            word_count += len(
                cell['source'].replace('#', '').lstrip().split(' '))

    return word_count


def add_arrays(*arrays: Tuple[np.ndarray]) -> np.ndarray:
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


softmax = np.logaddexp


def softmin(a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
    return - np.logaddexp(-a, -b, **kwargs)


def update_missing(d1, d2):
    """updates dictionary d1 with values from d2 if the
    keys aren't present in d2
    """
    d1.update((k, val) for k, val in d2.items() if k not in d1)


def piecewise(*arrays: Tuple[np.ndarray],
              condlist: List[np.ndarray],
              funclist: List[Callable],
              out: Optional[np.ndarray] = None, **kwargs: Dict[str, Any]):
    """
    evaluates a piecewise-defined function, extends numpy.piecewise
    for multiple arguments

    Parameters
    ----------
    arrays : tuple[Union[ndarray, scalar]]
        the arrays overwhich to calculate the piecewise functions
        these arrays/scalars will all be broadcast to the same size.
    condlist : list of bool arrays or bool scalars
        Each boolean array corresponds to a function in `funclist`.  Wherever
        `condlist[i]` is True, `funclist[i](x)` is used as the output value.
        Each boolean array in `condlist` selects a piece of `x`,
        and should therefore be of the same shape as `x`.
        The length of `condlist` must correspond to that of `funclist`.
        If one extra function is given, i.e. if
        ``len(funclist) == len(condlist) + 1``, then that extra function
        is the default value, used wherever all conditions are false.
    funclist : list of callables, f(*arrays,**kwargs), or scalars
        Each function is evaluated over `x` wherever its corresponding
        condition is True.  It should take a 1d array as input and give an 1d
        array or a scalar value as output.  If, instead of a callable,
        a scalar is provided then a constant function (``lambda x: scalar``) is
        assumed.
    kwargs : dict, optional
        Keyword arguments used in calling `piecewise` are passed to the
        functions upon execution, i.e., if called
        ``piecewise(..., ..., alpha=1)``, then each function is called as
        ``f(x, alpha=1)``.

    Returns
    -------
    out : ndarray
        The output is the same shape and type as x and is found by
        calling the functions in `funclist` on the appropriate portions of `x`,
        as defined by the boolean arrays in `condlist`.  Portions not covered
        by any condition have a default value of 0.

    Notes
    -----

        The result is::
            |--
            |funclist[0](arrays[0][condlist[0]], arrays[1][condlist[1]]...)
      out = |funclist[1](arrays[0][condlist[1]], arrays[1][condlist[1]]...)
            |...
            |funclist[n2](arrays[0][condlist[n2]], arrays[1][condlist[n2]]...)
            |--

    """
    n2 = len(funclist)
    arrays = np.broadcast_arrays(*arrays)

    if out is None:
        out = np.empty_like(arrays[0])

    n = len(condlist)
    if n == n2 - 1:  # compute the "otherwise" condition.
        condelse = ~np.any(condlist, axis=0, keepdims=True)
        condlist = np.concatenate([condlist, condelse], axis=0)
        n += 1
    elif n != n2:
        raise ValueError(
            "with {} condition(s), either {} or {} functions are expected"
                .format(n, n, n + 1))

    for k in range(n):
        item = funclist[k]
        cond = condlist[k]
        if cond.any():
            if not callable(item):
                out[cond] = item
            else:
                vals = (arr[cond] for arr in arrays)
                out[cond] = item(*vals, **kwargs)
    return out


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


def _log_hyp2f1_a1_zgte(z: np.ndarray, e: np.ndarray, l: np.ndarray
                        ) -> np.ndarray:
    """
    Calculates asymptotic expantion of
    log Hypegeometric2F1(1, e * l, l, z)

    for 0 < z < 1  & z >= 1/e
    """
    z = np.asanyarray(z)

    log = np.log
    logz = log(z)
    e1 = e - 1
    loge1 = log(e1)
    loge = log(e)
    log1z = np.log1p(-z)

    return (
            0.5 * log(2 * np.pi)
            + logz
            - log1z
            + loge1 / 2
            + loge / 2
            + l * (
                    - logz
                    + (e - 1) * (loge1 - log1z)
                    - e * loge)
            + log(l) / 2)


def _log_hyp2f1_a1_zlte(z: np.ndarray, e: np.ndarray, l: np.ndarray
                        ) -> np.ndarray:
    """
    Calculates asymptotic expantion of
    log Hypegeometric2F1(1, e * l, l, z)

    for 0 < z < 1 & z < 1/e
    """

    z = np.asanyarray(z)

    ze = z * e
    return -np.log1p(-z * e)


_log_hyp2f1_a1_funcs = [_log_hyp2f1_a1_zlte, _log_hyp2f1_a1_zgte]


def _log_hyp2f1_a1(z: np.ndarray, e: np.ndarray, l: np.ndarray
                   ) -> np.ndarray:
    """
    Calculates asymptotic expantion of
    log Hypegeometric2F1(1, e * l, l, z)

    for 0 < z < 1
    """
    isscalar = all(np.isscalar(x) for x in (z, e, l))
    assert np.all(z < 1)
    assert np.all(0 < z)

    if isscalar:
        return _log_hyp2f1_a1_funcs[z * e > 1](z, e, l)
    else:
        z, e, l = np.broadcast_arrays(z, e, l)
        condition = z * e < 1
        hyp2f1 = piecewise(
            z, e, l, condlist=[condition],
            funclist=_log_hyp2f1_a1_funcs)
        return np.reshape(hyp2f1, z.shape)


def log_hyp2f1_a1(b: np.ndarray, c: np.ndarray, z: np.ndarray
                  ) -> np.ndarray:
    """
    Calculates asymptotic expantion of
    log Hypergeometric2F1(1, b, c, z)

    for 0 < z < 1
    """
    return _log_hyp2f1_a1(z, b / c, c)


def approx_hyp2f1_a1(b: np.ndarray, c: np.ndarray, z: np.ndarray
                     ) -> np.ndarray:
    """
    Calculates an approximate asymptotic expansion of
    Hypergeometric2F1(1, b, c, z)

    for 0 < z < 1
    """
    return np.exp(_log_hyp2f1_a1(z, b / c, c))


def hyp2f1_a1(b: np.ndarray, c: np.ndarray, z: np.ndarray, *,
              APPROX_B: float = 30.) -> np.ndarray:
    """
    calculates the value of Hypergeometric2F1(1, b, c, z)
    in a more numerically stable manner

    for b > c > 0

    when b > 30 the hypergeometric function provided by
    scipy can be numerically unstable, so we use the asymptotic
    expansions provided by Cvitković et al. [0] to evaluate the value of the
    hypergeometric functions


    [0] Cvitković, Mislav, Ana-Sunčana Smith, and Jayant Pande.
        "Asymptotic Expansions of the Hypergeometric Function
        with Two Large Parameters—application to the Partition
        Function of a Lattice Gas in a Field of Traps."
        Journal of Physics A: Mathematical and Theoretical
        50.26 (2017): 265206.
        arXiv link: https://arxiv.org/abs/1602.05146
    """
    isscalar = all(np.isscalar(x) for x in (b, c, z))

    assert np.all(b > 0)
    assert np.all(c > 0)

    ## scipy function can be numerically unstable above a1 + a2 > 0
    if isscalar:
        cond = b < APPROX_B
        h = special.hyp2f1(1, b, c, z) if cond else approx_hyp2f1_a1(b, c, z)
    else:
        ## Calculate hyperbolic function
        b, c, z = np.broadcast_arrays(b, c, z)
        cond = b < APPROX_B

        h = np.empty_like(z)
        sel = cond
        if sel.any():
            h[sel] = special.hyp2f1(1, b[sel], c[sel], z[sel])

        sel = ~(cond & np.isfinite(h))
        if sel.any():
            h[sel] = approx_hyp2f1_a1(b[sel], c[sel], z[sel])

    return h