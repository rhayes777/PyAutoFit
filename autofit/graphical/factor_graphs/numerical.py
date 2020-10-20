from typing import Tuple, Dict, NamedTuple, Optional

import numpy as np

from autofit.mapper.variable import Variable

def numerical_func_jacobian(
        factor: "AbstractNode",
        values: Dict[Variable, np.array],
        variables: Optional[Tuple[Variable, ...]],
        _eps: float = 1e-6,
        _calc_deterministic: bool = True,
) -> Tuple["FactorValue", "JacobianValue"]:
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
    >>> x_ = Variable('x')
    >>> y_ = Variable('y')
    >>> A = np.arange(4).reshape(2, 2)
    >>> dot = factor(lambda x: A.dot(x))(x_) == y_
    >>> dot.jacobian([x_], {x_: [1, 2]})
    JacobianValue(
        log_value={'x': array([[0.], [0.]])},
        deterministic_values={('x', 'y'): array([[0., 2.], [1., 3.]])})
    """
    if variables is None:
        variables = factor.variables
    # copy the input array
    p0 = {v: np.array(x, dtype=float) for v, x in values.items()}
    f0 = factor(p0)
    log_f0 = f0.log_value
    det_vars0 = f0.deterministic_values

    jac_f = {
        v: np.empty(np.shape(values[v]) + np.shape(log_f0))
        for v in variables}
    if _calc_deterministic:
        jac_det = {
            (det, v): np.empty(
                np.shape(val) + np.shape(values[v]))
            for v in variables
            for det, val in det_vars0.items()}
        det_slices = {
            v: (slice(None),) * np.ndim(a) for v, a in values.items()}
    else:
        jac_det = {}

    for v in variables:
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

    return f0, JacobianValue(jac_f, jac_det)


def numerical_func_jacobian_hessian(
        factor: "AbstractNode",
        values: Dict[Variable, np.array],
        variables: Optional[Tuple[Variable, ...]] = None,
        _eps: float = 1e-6,
        _calc_deterministic: bool = True
) -> Tuple["FactorValue", "JacobianValue", "HessianValue"]:
    
    if variables is None:
        variables = factor.variables
    p0 = {v: np.array(x, dtype=float) for v, x in values.items()}
    f0, jac_f0 = factor.func_jacobian(p0, variables)
    (log_f0, det_vars0), (grad_f0, jac_det_vars0) = f0, jac_f0

    f_shape = np.shape(log_f0)
    f_size = np.prod(f_shape)
    hess_f = {
        v: np.empty(np.shape(values[v]) * 2 + f_shape)
        for v in variables}

    if _calc_deterministic:
        det_shapes = {v: d.shape for v, d in det_vars0.items()}
        for v in det_vars0:
            hess_f[v] = 0.

    for v in variables:
        x0 = p0[v]
        if x0.shape:
            inds = tuple(a.ravel() for a in np.indices(x0.shape))
            for ind in zip(*inds):
                x0[ind] += _eps
                p0[v] = x0
                grad_f, _ = factor.jacobian(
                    p0, (v,), _calc_deterministic=False)
                x0[ind] -= _eps
                hess_f[v][ind] = grad_f[v] - grad_f0[v]
                
            # Symmetrise Hessian
            triu = np.triu_indices(x0.size, 1)
            i = tuple(ind[triu[0]] for ind in inds)
            j = tuple(ind[triu[1]] for ind in inds)
            np.add.at(hess_f[v], i + j, hess_f[v][j + i])
            hess_f[v][i + j] += hess_f[v][j + i]
            hess_f[v][i + j] /= 2 
            hess_f[v][j + i] = hess_f[v][i + j]
            
            if _calc_deterministic:
                var_size = x0.size
                hess = hess_f[v].reshape((var_size, var_size, f_size))
                for d, d_shape in det_shapes.items():
                    jac = jac_det_vars0[d, v].reshape(np.prod(d_shape), var_size)
                    hess_d = np.einsum(
                        "ij,jkl,mk->iml", jac, hess, jac)
                    hess_f[d] += hess_d.reshape(d_shape + d_shape + f_shape)
            
        else:
            p0[v] += _eps
            grad_f, _ = factor.jacobian(
                p0, (v,), _calc_deterministic=False)
            p0[v] -= _eps
            hess_f[v] = grad_f[v] - grad_f0[v]
            
            if _calc_deterministic:
                hess = hess_f[v].reshape(1, 1, f_size)
                for d, d_shape in det_shapes.items():
                    jac = jac_det_vars0[d, v].reshape(np.prod(d_shape))
                    hess_f[d] += (
                        jac[:, None, None] 
                        * jac[None, :, None] * hess[None, None, :])

    return f0, jac_f0, hess_f

# Import these at the end to resolve circular imports and typing
from .abstract import (
    FactorValue, JacobianValue, HessianValue)