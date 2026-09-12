"""
Regenerate the committed model-figure evidence renders.

Run from the PyAutoFit repo root, with the worktree environment active::

    python docs/images/model_figures/make_figures.py

Every PNG in this directory is written by this script at ``width=14.0`` inches
and DPI 100, so none exceeds the 1400 px width budget. The figures are the
acceptance evidence for the ``model-figures`` epic's phase 2 and the images the
cookbook docs pages embed.

Most models here are lifted from the PyAutoFit cookbooks (``autofit_workspace``
``scripts/cookbooks/{model,multi_level_model,multiple_datasets}.py``) so the
figure a reader sees in the docs is the model the surrounding code composes.

The three lens figures (``simple_lens``, ``mge_pixelized``, ``group_scale``) are
built from ``test_autofit/graph_spec/lens_doubles.py``. Those are **structural
doubles** -- classes with the same constructor signatures, priors and
composition as the real PyAutoLens components, so PyAutoFit can render them
without depending on PyAutoLens. Phase 3 of the epic renders the real models and
replaces them.
"""

import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import autofit as af

OUTPUT_PATH = Path(__file__).resolve().parent
WIDTH = 14.0


def write(model, name: str, **kwargs):
    """Render one model to ``<name>.png`` in this directory."""
    af.ModelPlotter(model).figure(
        path=str(OUTPUT_PATH),
        filename=name,
        format="png",
        width=WIDTH,
        **kwargs,
    )
    print(f"wrote {name}.png")


# -- the cookbook models ----------------------------------------------------


def gaussian():
    """``scripts/cookbooks/model.py`` -- the plain ``af.Model``."""
    return af.Model(af.ex.Gaussian)


def gaussian_customised():
    """
    ``scripts/cookbooks/model.py`` -- "Model Customization (Model)".

    ``centre`` is fixed, ``normalization`` is given a new prior and ``sigma``
    keeps the prior its configuration file supplies.
    """
    model = af.Model(af.ex.Gaussian)
    model.centre = 0.0
    model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
    return model


def gaussian_collection():
    """
    ``scripts/cookbooks/model.py`` -- "Model Customization (Collection)".
    """
    gaussian = af.Model(af.ex.Gaussian)
    gaussian.normalization = 1.0
    gaussian.sigma = af.GaussianPrior(mean=0.0, sigma=1.0)

    exponential = af.Model(af.ex.Exponential)
    exponential.centre = 50.0
    exponential.add_assertion(exponential.rate > 5.0)

    return af.Collection(gaussian=gaussian, exponential=exponential)


class MultiLevelGaussian:
    def __init__(self, normalization=1.0, sigma=5.0):
        self.normalization = normalization
        self.sigma = sigma


class MultiLevelGaussians:
    def __init__(
        self,
        higher_level_centre: float = 50.0,
        gaussian_list: List[MultiLevelGaussian] = None,
    ):
        self.higher_level_centre = higher_level_centre
        self.gaussian_list = gaussian_list


def multi_level():
    """
    ``scripts/cookbooks/multi_level_model.py`` -- the multi-level composition:
    two ``Gaussian``s whose centre is a parameter of the level above them.
    """
    return af.Model(
        MultiLevelGaussians,
        gaussian_list=[af.Model(MultiLevelGaussian), af.Model(MultiLevelGaussian)],
    )


class _Analysis(af.Analysis):
    """
    A do-nothing analysis. ``AnalysisFactor`` only needs an analysis object to
    pair with the model; nothing here is ever fitted.

    It carries a ``data`` attribute because that is what makes the figure draw
    an **observed** node for the dataset: observed data and a fixed model
    constant are different concepts and must not share the grey pill.
    """

    def __init__(self, index):
        self.index = index
        self.data = [0.0, 1.0, 2.0]

    def log_likelihood_function(self, instance):
        return 0.0


def _factor_graph(model_list):
    return af.FactorGraphModel(
        *[
            af.AnalysisFactor(prior_model=model, analysis=_Analysis(index))
            for index, model in enumerate(model_list)
        ]
    )


def _multiple_datasets_model():
    model = af.Model(af.ex.Gaussian)
    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
    model.sigma = af.GaussianPrior(mean=10.0, sigma=5.0)
    return model


def multiple_datasets_shared():
    """
    ``scripts/cookbooks/multiple_datasets.py`` -- one model shared by all three
    datasets.
    """
    model = _multiple_datasets_model()
    return _factor_graph([model, model, model]).global_prior_model


def multiple_datasets_variable():
    """
    ``scripts/cookbooks/multiple_datasets.py`` -- a shared ``centre`` with a
    ``normalization`` and ``sigma`` free per dataset.
    """
    model = _multiple_datasets_model()

    model_list = []
    for _ in range(3):
        model_analysis = model.copy()
        model_analysis.normalization = af.LogUniformPrior(
            lower_limit=1e-2, upper_limit=1e2
        )
        model_analysis.sigma = af.GaussianPrior(mean=10.0, sigma=5.0)
        model_list.append(model_analysis)

    return _factor_graph(model_list).global_prior_model


def multiple_datasets_relational():
    """
    ``scripts/cookbooks/multiple_datasets.py`` -- ``sigma`` follows the relation
    ``sigma_m * x + sigma_c`` across the datasets, so adding datasets adds no
    parameters.
    """
    model = af.Collection(gaussian=af.Model(af.ex.Gaussian))

    sigma_m = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)
    sigma_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

    model_list = []
    for x in [1.0, 2.0, 3.0]:
        model_analysis = model.copy()
        model_analysis.gaussian.sigma = (sigma_m * x) + sigma_c
        model_list.append(model_analysis)

    return _factor_graph(model_list).global_prior_model


# -- the graphical models ---------------------------------------------------


def _graphical_model():
    """One dataset's ``Gaussian``, with a centre that a hyper-prior can draw."""
    model = af.Model(af.ex.Gaussian)
    model.centre = af.TruncatedGaussianPrior(
        mean=50.0, sigma=20.0, lower_limit=0.0, upper_limit=100.0
    )
    model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
    model.sigma = af.GaussianPrior(mean=10.0, sigma=5.0)
    return model


def graphical_shared():
    """
    Plate notation, **shared**: one ``centre``, ``normalization`` and ``sigma``
    for all three datasets -- literally the same three numbers.
    """
    model = _graphical_model()
    return _factor_graph([model, model, model]).global_prior_model


def graphical_variable():
    """
    Plate notation, **partly shared**: one ``centre`` for every dataset, with a
    ``normalization`` and ``sigma`` of its own per dataset.
    """
    model = _graphical_model()

    model_list = []
    for _ in range(3):
        model_analysis = model.copy()
        model_analysis.normalization = af.LogUniformPrior(
            lower_limit=1e-2, upper_limit=1e2
        )
        model_analysis.sigma = af.GaussianPrior(mean=10.0, sigma=5.0)
        model_list.append(model_analysis)

    return _factor_graph(model_list).global_prior_model


def graphical_hierarchical():
    """
    Plate notation, **hierarchical**: three *different* centres, each drawn from
    one parent ``GaussianPrior`` whose ``mean`` and ``sigma`` are themselves
    free (HowToFit chapter 3, tutorial 4).

    The teaching pair of the epic: put this figure beside ``graphical_shared``
    and the difference between "the same number" and "from a common population"
    is the difference between a blue reference and a violet arrow.
    """
    model_list = [_graphical_model() for _ in range(3)]

    hierarchical_factor = af.HierarchicalFactor(
        af.GaussianPrior,
        mean=af.TruncatedGaussianPrior(
            mean=50.0, sigma=10, lower_limit=0.0, upper_limit=100.0
        ),
        sigma=af.TruncatedGaussianPrior(
            mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=100.0
        ),
    )
    for model in model_list:
        hierarchical_factor.add_drawn_variable(model.centre)

    factor_graph = af.FactorGraphModel(
        *[
            af.AnalysisFactor(prior_model=model, analysis=_Analysis(index))
            for index, model in enumerate(model_list)
        ],
        hierarchical_factor,
    )
    return factor_graph.global_prior_model


def composite_shared_relation_assertion():
    """
    All three of sharing, relation and assertion at once -- acceptance case 3.
    """
    model = af.Collection(
        a=af.Model(af.ex.Gaussian),
        b=af.Model(af.ex.Gaussian),
    )
    model.b.centre = model.a.centre
    model.b.sigma = model.a.sigma * 2.0
    model.add_assertion(model.a.sigma > 5.0)
    return model


# -- the lens structural doubles --------------------------------------------


def lens_models():
    from test_autofit.graph_spec.lens_doubles import (
        simple_lens_model,
        mge_model,
        group_model,
    )

    return {
        "simple_lens": simple_lens_model(),
        "mge_pixelized": mge_model(),
        "group_scale": group_model(),
    }


FIGURES = {
    "gaussian": gaussian,
    "gaussian_customised": gaussian_customised,
    "gaussian_collection": gaussian_collection,
    "multi_level": multi_level,
    "multiple_datasets_shared": multiple_datasets_shared,
    "multiple_datasets_variable": multiple_datasets_variable,
    "multiple_datasets_relational": multiple_datasets_relational,
    "composite_shared_relation_assertion": composite_shared_relation_assertion,
    "graphical_shared": graphical_shared,
    "graphical_variable": graphical_variable,
    "graphical_hierarchical": graphical_hierarchical,
}


def main():
    for name, builder in FIGURES.items():
        write(builder(), name)

    for name, model in lens_models().items():
        write(model, name)


if __name__ == "__main__":
    main()
