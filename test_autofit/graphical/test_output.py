import pytest

import autofit as af
from autoconf.conf import with_config
from autofit import graphical as g
from autofit.graphical import AnalysisFactor, PriorFactor
from autofit.graphical.declarative.graph import DeclarativeGraphOutput
from autofit.mock.mock import MockAnalysis
from autofit.tools.namer import namer

MAX_STEPS = 3


@pytest.fixture(
    autouse=True
)
def reset_namer():
    namer.reset()
    yield
    namer.reset()


class MockResult(af.MockResult):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def projected_model(self):
        return self.model


class MockSearch(af.MockSearch):
    def fit(
            self,
            model,
            analysis,
            info=None,
            pickle_files=None,
            log_likelihood_cap=None
    ):
        super().fit(
            model,
            analysis,
        )
        return MockResult(model)


@pytest.fixture(
    name="factor_graph_model"
)
def make_factor_graph_model():
    model_factor_1 = g.AnalysisFactor(
        af.Collection(
            one=af.UniformPrior()
        ),
        MockAnalysis()
    )
    model_factor_2 = g.AnalysisFactor(
        af.Collection(
            one=af.UniformPrior()
        ),
        MockAnalysis()
    )

    return g.FactorGraphModel(
        model_factor_1,
        model_factor_2
    )


def test_non_trivial_model():
    one = af.Model(af.Gaussian)
    two = af.Model(af.Gaussian)

    one.centre = two.centre

    model_factor_1 = g.AnalysisFactor(
        one,
        MockAnalysis()
    )
    model_factor_2 = g.AnalysisFactor(
        two,
        MockAnalysis()
    )

    info = g.FactorGraphModel(
        model_factor_1,
        model_factor_2
    ).graph.info

    assert info == """PriorFactors

PriorFactor0 (AnalysisFactor0.intensity)                                                  UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor1 (AnalysisFactor0.sigma)                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor2 (AnalysisFactor0.centre, AnalysisFactor1.centre)                             UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor3 (AnalysisFactor1.intensity)                                                  UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor4 (AnalysisFactor1.sigma)                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0

AnalysisFactors

AnalysisFactor0

centre (AnalysisFactor1.centre, PriorFactor2)                                             UniformPrior, lower_limit = 0.0, upper_limit = 1.0
intensity (PriorFactor0)                                                                  UniformPrior, lower_limit = 0.0, upper_limit = 1.0
sigma (PriorFactor1)                                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0

AnalysisFactor1

centre (AnalysisFactor0.centre, PriorFactor2)                                             UniformPrior, lower_limit = 0.0, upper_limit = 1.0
intensity (PriorFactor3)                                                                  UniformPrior, lower_limit = 0.0, upper_limit = 1.0
sigma (PriorFactor4)                                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""


def _run_optimisation(
        factor_graph_model
):
    factor_graph_model.optimise(
        MockSearch(),
        max_steps=MAX_STEPS,
        name="name",
        log_interval=1,
        visualise_interval=1,
        output_interval=1,
    )


@pytest.fixture(
    name="factor_graph"
)
def make_factor_graph(
        factor_graph_model
):
    return factor_graph_model.graph


def test_factors_with_type(
        factor_graph
):
    factor_type = AnalysisFactor
    factors = factor_graph._factors_with_type(
        factor_type
    )
    assert len(factors) == 2
    for factor in factors:
        assert isinstance(
            factor,
            AnalysisFactor
        )


def test_factors_grouped_by_type(
        factor_graph
):
    factors_by_type = factor_graph.factors_by_type()

    assert len(factors_by_type) == 2
    assert len(factors_by_type[AnalysisFactor]) == 2
    assert len(factors_by_type[PriorFactor]) == 2


@pytest.fixture(
    name="prior_factor"
)
def make_prior_factor(
        factor_graph
):
    return factor_graph.prior_factors[0]


@pytest.fixture(
    name="analysis_factor"
)
def make_analysis_factor(
        factor_graph
):
    return factor_graph.analysis_factors[0]


@pytest.fixture(
    name="declarative_graph_output"
)
def make_declarative_graph_output(
        factor_graph
):
    return DeclarativeGraphOutput(
        factor_graph
    )


def test_info_for_prior_factor(
        declarative_graph_output,
        prior_factor
):
    assert declarative_graph_output.info_for_prior_factor(
        prior_factor
    ) == "PriorFactor0 (AnalysisFactor0.one)                                                        UniformPrior, lower_limit = 0.0, upper_limit = 1.0"


def test_info_for_analysis_factor(
        declarative_graph_output,
        analysis_factor
):
    assert declarative_graph_output.info_for_analysis_factor(
        analysis_factor
    ) == """AnalysisFactor0

one (PriorFactor1)                                                                        UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""


def test_related_factors(
        factor_graph,
        prior_factor
):
    assert len(factor_graph.related_factors(
        list(prior_factor.variables)[0]
    )) == 2


def test_graph_info(
        factor_graph
):
    assert factor_graph.info == """PriorFactors

PriorFactor0 (AnalysisFactor0.one)                                                        UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor1 (AnalysisFactor1.one)                                                        UniformPrior, lower_limit = 0.0, upper_limit = 1.0

AnalysisFactors

AnalysisFactor0

one (PriorFactor0)                                                                        UniformPrior, lower_limit = 0.0, upper_limit = 1.0

AnalysisFactor1

one (PriorFactor1)                                                                        UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""


@with_config(
    "general",
    "output",
    "remove_files",
    value=False
)
def test_output(
        output_directory,
        factor_graph_model
):
    factor_graph_model.model_factors[0]._name = "factor_1"
    factor_graph_model.model_factors[1]._name = "factor_2"
    _run_optimisation(factor_graph_model)

    path = output_directory / "name/factor_1"

    assert path.exists()
    assert (output_directory / "name/factor_2").exists()

    for number in range(MAX_STEPS):
        assert (path / f"optimization_{number}").exists()


@with_config(
    "general",
    "output",
    "remove_files",
    value=False
)
def test_default_output(
        output_directory,
        factor_graph_model
):
    _run_optimisation(factor_graph_model)
    assert (output_directory / "name/AnalysisFactor0").exists()
    assert (output_directory / "name/AnalysisFactor1").exists()
