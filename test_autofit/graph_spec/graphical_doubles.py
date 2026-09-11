"""
The four graphical models the phase-4 figures are asserted against.

Three datasets each, so a plate can form, and one ``_Analysis`` per dataset that
carries ``data`` -- which is what makes the extraction emit an ``observed`` row.
They mirror ``docs/images/model_figures/make_figures.py``'s ``graphical_*``
builders (the committed evidence PNGs) and the ``autofit_workspace``
``multiple_datasets`` cookbook, so a test failure and a figure regression are
the same failure.
"""

import autofit as af


class DataAnalysis(af.Analysis):
    """An analysis that carries ``data`` -- and does nothing else."""

    def __init__(self, index):
        self.index = index
        self.data = [0.0, 1.0, 2.0]

    def log_likelihood_function(self, instance):
        return 0.0


def dataset_model():
    """One dataset's ``Gaussian``, with a centre a hyper-prior can draw."""
    model = af.Model(af.ex.Gaussian)
    model.centre = af.TruncatedGaussianPrior(
        mean=50.0, sigma=20.0, lower_limit=0.0, upper_limit=100.0
    )
    model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
    model.sigma = af.GaussianPrior(mean=10.0, sigma=5.0)
    return model


def factor_graph(model_list, analysis_cls=DataAnalysis):
    return af.FactorGraphModel(
        *[
            af.AnalysisFactor(prior_model=model, analysis=analysis_cls(index))
            for index, model in enumerate(model_list)
        ]
    )


def shared_graph(analysis_cls=DataAnalysis):
    """One model object for all three datasets: every prior is literally shared."""
    model = dataset_model()
    return factor_graph([model, model, model], analysis_cls)


def variable_graph():
    """A shared ``centre``; a ``normalization`` and ``sigma`` per dataset."""
    model = dataset_model()
    model_list = []
    for _ in range(3):
        copy = model.copy()
        copy.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
        copy.sigma = af.GaussianPrior(mean=10.0, sigma=5.0)
        model_list.append(copy)
    return factor_graph(model_list)


def relational_graph():
    """``sigma`` follows ``sigma_m * x + sigma_c``, so datasets add no parameters."""
    model = af.Collection(gaussian=dataset_model())
    sigma_m = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)
    sigma_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

    model_list = []
    for x in [1.0, 2.0, 3.0]:
        copy = model.copy()
        copy.gaussian.sigma = (sigma_m * x) + sigma_c
        model_list.append(copy)
    return factor_graph(model_list)


def hierarchical_graph():
    """
    Three *different* centres, each drawn from one parent ``GaussianPrior``
    whose own ``mean`` and ``sigma`` are free (HowToFit chapter 3, tutorial 4).
    """
    model_list = [dataset_model() for _ in range(3)]

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

    return af.FactorGraphModel(
        *[
            af.AnalysisFactor(prior_model=model, analysis=DataAnalysis(index))
            for index, model in enumerate(model_list)
        ],
        hierarchical_factor,
    )
