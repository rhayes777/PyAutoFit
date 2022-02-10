import numpy as np
from scipy import special


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
    for _ in range(4):
        f0 = psilog(x0) - c
        x0 = x0 - f0 / grad_psilog(x0)

    return x0
