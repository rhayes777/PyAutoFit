"""
The EP factor-graph figure -- the *diagnostic* view of a graphical fit.

``autofit.model_figure`` draws a model's containment structure; this package
draws the **factor graph** an :class:`~autofit.graphical.EPOptimiser` actually
sweeps, with the run's state painted onto it.  It follows the same three-layer
split, each layer importable and testable on its own:

* :mod:`autofit.model_figure.ep.spec` -- structure: which factor nodes, which
  variable nodes, which edges, which plates.
* :mod:`autofit.model_figure.ep.state` -- overlay: what the EP run did to each
  factor (updates, sweeps, age, reverted variables).
* ``presentation`` / ``layout`` / ``render`` -- what is drawn, where it goes,
  and matplotlib (phase-5 steps 3 onwards).

Nothing here imports a drawing library at module level, so ``import autofit``
stays drawing-free.
"""

__all__ = []
