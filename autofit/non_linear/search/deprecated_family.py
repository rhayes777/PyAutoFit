"""
The deprecation hook of the former family bases (search-extensibility A4c, D14).

``AbstractMCMC``, ``AbstractNest`` and ``AbstractMLE`` are no longer in any public
search's MRO. Their modules still export them, as distinct thin subclasses of
``NonLinearSearch`` that keep the former constructor defaults and plots, so external
subclasses keep working for one release; subclassing one warns.
"""

import warnings


def warn_family_subclass(cls, family: str, kind: str):
    warnings.warn(
        f"{cls.__name__} subclasses {family}, which is deprecated and will be removed "
        f"one release after search-extensibility phase A4. Subclass "
        f"`autofit.NonLinearSearch` directly, declare `posterior_kind = "
        f"PosteriorKind.{kind}` and apply that kind's defaults in `__init__` "
        f"(`autofit.non_linear.search.posterior_kind.{kind}`).",
        FutureWarning,
        stacklevel=3,
    )
