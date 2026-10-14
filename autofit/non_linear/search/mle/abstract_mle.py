from abc import ABC

from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.search.deprecated_family import warn_family_subclass
from autofit.non_linear.search.posterior_kind import POINT


class AbstractMLE(NonLinearSearch, ABC):
    """
    **Deprecated** (search-extensibility A4c): no public search inherits it any more,
    and it is removed one release later. Subclass ``NonLinearSearch`` with
    ``posterior_kind = PosteriorKind.POINT`` and apply
    ``autofit.non_linear.search.posterior_kind.POINT`` in ``__init__`` instead.

    Kept as a distinct class with the former defaults (``InitializerBall(0.49, 0.51)``,
    a ``clipper`` defaulting to ``ClipperNone``, the MLE trace plots), so an existing
    subclass behaves exactly as before; subclassing it warns.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warn_family_subclass(cls, "AbstractMLE", "POINT")

    def __init__(self, initializer=None, clipper=None, **kwargs):
        self.clipper = POINT.clipper_for(clipper)

        super().__init__(initializer=POINT.initializer_for(initializer), **kwargs)

    def plot_results(self, samples):
        POINT.plot_results(self, samples)
