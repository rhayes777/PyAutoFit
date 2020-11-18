from typing import Tuple, Dict, NamedTuple, Optional, Union
from functools import partial

import numpy as np

from autofit.mapper.variable import Variable
from autofit.graphical.utils import aggregate, Axis
from autofit.graphical.factor_graphs.abstract import \
    FactorValue, JacobianValue, HessianValue

def numerical_func_jacobian(
        factor: "AbstractNode",
        values: Dict[Variable, np.array],
        variables: Optional[Tuple[Variable, ...]] = None,
        axis: Axis = False, 
        _eps: float = 1e-6,
        _calc_deterministic: bool = True,
) -> Tuple[FactorValue, JacobianValue]:
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
    f0 = factor(p0, axis=axis)
    log_f0 = f0.log_value
    det_vars0 = f0.deterministic_values

    fjac = {
        v: FactorValue(np.empty(np.shape(log_f0) + np.shape(values[v])))
        for v in variables}
    if _calc_deterministic:
        for v, grad in fjac.items():
            grad.deterministic_values = {
                det: np.empty(np.shape(val) + np.shape(values[v]))
                for det, val in det_vars0.items()
            }
        det_slices = {
            v: (slice(None),) * np.ndim(a) for v, a in det_vars0.items()}

    for v, grad in fjac.items():
        x0 = p0[v]
        v_jac = grad.deterministic_values
        if x0.shape:
            inds = tuple(a.ravel() for a in np.indices(x0.shape))
            i0 = tuple(slice(None) for _ in range(np.ndim(log_f0)))
            for ind in zip(*inds):
                x0[ind] += _eps
                p0[v] = x0
                f = factor(p0, axis=axis)
                x0[ind] -= _eps

                # print(ind)
                grad[i0 + ind] = (f - f0) / _eps
                if _calc_deterministic:
                    det_vars = f.deterministic_values
                    for det, val in det_vars.items():
                        v_jac[det][det_slices[det] + ind] = \
                            (val - det_vars0[det]) / _eps
        else:
            p0[v] += _eps
            f = factor(p0, axis=axis)
            p0[v] -= _eps

            grad.itemset((f - f0) / _eps)
            if _calc_deterministic:
                det_vars = f.deterministic_values
                for det, val in det_vars.items():
                    v_jac[det] = (val - det_vars0[det]) / _eps

    return f0, fjac


def numerical_func_jacobian_hessian(
        factor: "AbstractNode",
        values: Dict[Variable, np.array],
        variables: Optional[Tuple[Variable, ...]] = None,
        axis: Optional[Union[bool, int, Tuple[int, ...]]] = False, 
        _eps: float = 1e-6,
        _calc_deterministic: bool = True
) -> Tuple[FactorValue, JacobianValue, HessianValue]:
    
    if variables is None:
        variables = factor.variables

    # agg = partial(aggregate, axis=axis)

    p0 = {v: np.array(x, dtype=float) for v, x in values.items()}
    f0, fjac0 = factor.func_jacobian(p0, variables, axis=axis)
    # grad_f0, jac_det_vars0 = jac_f0

    log_f0 = f0.log_value
    det_vars0 = f0.deterministic_values
    f_shape = np.shape(log_f0)
    f_size = np.prod(f_shape, dtype=int)
    fhess0 = {
        v: np.empty(f_shape + np.shape(values[v]) * 2)
        for v in variables}

    if _calc_deterministic:
        det_shapes = {v: d.shape for v, d in det_vars0.items()}
        for v in det_vars0:
            fhess0[v] = 0.

    for v, hess in fhess0.items():
        x0 = p0[v]
        if x0.shape:
            inds = tuple(a.ravel() for a in np.indices(x0.shape))
            i0 = tuple(slice(None) for _ in f_shape)
            for ind in zip(*inds):
                x0[ind] += _eps
                p0[v] = x0
                fjac1 = factor.jacobian(
                    p0, (v,), axis=axis, _eps=_eps, _calc_deterministic=False)
                x0[ind] -= _eps
                hess[i0 + ind] = (fjac1[v] - fjac0[v])/_eps
                
            # Symmetrise Hessian
            triu = np.triu_indices(x0.size, 1) # indices of upper diagonal
            i = tuple(ind[triu[0]] for ind in inds)
            j = tuple(ind[triu[1]] for ind in inds)
            upper = i0 + i + j
            lower = i0 + j + i
            hess[upper] += hess[lower]
            hess[upper] /= 2 
            hess[lower] = hess[upper]
            
            if _calc_deterministic:
                var_size = x0.size
                if f_shape:
                    hess2d = hess.reshape(f_size, var_size, var_size)
                    for d, d_shape in det_shapes.items():
                        jac = fjac0[v][d].reshape(
                            np.prod(d_shape), var_size)
                        hess_d = np.einsum(
                            "ij,ljk,mk->lim", jac, hess2d, jac)
                        fhess0[d] += hess_d.reshape(
                            d_shape + d_shape + f_shape)
                else:
                    hess2d = hess.reshape((var_size, var_size))
                    for d, d_shape in det_shapes.items():
                        jac = fjac0[v][d].reshape(
                            np.prod(d_shape), var_size)
                        hess_d = np.linalg.multi_dot(
                            [jac, hess2d, jac.T])
                        fhess0[d] += hess_d.reshape(d_shape + d_shape)
            
        else:
            p0[v] += _eps
            fjac1 = factor.jacobian(
                p0, (v,), axis=axis, _eps=_eps, _calc_deterministic=False)
            p0[v] -= _eps
            fhess0[v] = hess = (fjac1[v] - fjac0[v])/_eps
            
            if _calc_deterministic:
                hess2d = hess.reshape(f_size, 1, 1)
                for d, d_shape in det_shapes.items():
                    jac = fjac0[v][d].reshape(np.prod(d_shape))
                    fhess0[d] += (
                        jac[:, None, None] 
                        * jac[None, :, None] * hess[None, None, :])

    return f0, fjac0, fhess0
