from typing import Tuple, Dict, NamedTuple

import numpy as np

from autofit.mapper.variable import Variable


class JacobianValue(NamedTuple):
    log_value: Dict[str, np.ndarray]
    deterministic_values: Dict[Tuple[str, str], np.ndarray]


def numerical_jacobian(
        factor,
        args: Tuple[Variable, ...],
        kwargs: Dict[Variable, np.array],
        _eps: float = 1e-6,
        _calc_deterministic: bool = True,
) -> JacobianValue:
    """Calculates the numerical Jacobian of the passed factor

    the arguments passed correspond to the variable that we want
    to take the derivatives for

    the values must be passed as keywords

    _eps specifies the numerical accuracy of the numerical derivative

    returns a jac = JacobianValue namedtuple

    jac.log_value stores the Jacobian of the factor output as a dictionary
    where the keys are the names of the variables

    jac.determinisic_values store the Jacobian of the deterministic variables
    where the keys are a Tuple[str, str] pair of the variable and deterministic
    variables

    Example
    -------
    >>> import numpy as np
    >>> y_ = Variable('y')
    >>> A = np.arange(4).reshape(2, 2)
    >>> dot = factor(lambda x: A.dot(x))(x_) == y_
    >>> dot.jacobian('x', x=[1, 2])
    JacobianValue(
        log_value={'x': array([[0.], [0.]])},
        deterministic_values={('x', 'y'): array([[0., 2.], [1., 3.]])})
    """
    # copy the input array
    p0 = {v: np.array(x, dtype=float) for v, x in kwargs.items()}
    f0 = factor(p0)
    log_f0 = f0.log_value
    det_vars0 = f0.deterministic_values

    jac_f = {
        v: np.empty(np.shape(kwargs[v]) + np.shape(log_f0))
        for v in args}
    if _calc_deterministic:
        jac_det = {
            (det, v): np.empty(
                np.shape(val) + np.shape(kwargs[v]))
            for v in args
            for det, val in det_vars0.items()}
        det_slices = {
            v: (slice(None),) * np.ndim(a) for v, a in kwargs.items()}
    else:
        jac_det = {}

    for v in args:
        x0 = p0[v]
        if x0.shape:
            inds = tuple(a.ravel() for a in np.indices(x0.shape))
            for ind in zip(*inds):
                x0[ind] += _eps
                p0[v] = x0
                f = factor(p0)
                x0[ind] -= _eps

                jac_f[v][ind] = (f.log_value - log_f0) / _eps
                if _calc_deterministic:
                    det_vars = f.deterministic_values
                    for det, val in det_vars.items():
                        jac_det[det, v][det_slices[v] + ind] = \
                            (val - det_vars0[det]) / _eps
        else:
            p0[v] += _eps
            f = factor(p0)
            p0[v] -= _eps

            jac_f[v] = (f.log_value - log_f0) / _eps
            if _calc_deterministic:
                det_vars = f.deterministic_values
                for det, val in det_vars.items():
                    jac_det[det, v] = (val - det_vars0[det]) / _eps

    return JacobianValue(jac_f, jac_det)
