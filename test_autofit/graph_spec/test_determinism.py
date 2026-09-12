"""
Determinism and layer-purity of the semantic extraction.

Determinism is part of the epic's acceptance: the same model must serialise to
byte-identical output every time, including ordering, and the *same model built
again* (after an id reset) must too -- otherwise no figure can be diffed or
committed as evidence.
"""

import itertools
import json
import subprocess
import sys

import autofit as af
from autofit.example.model import PhysicalNFW
from autofit.graph_spec import GraphSpec
from autofit.tools.namer import namer

from .conftest import NullAnalysis


def _reset_ids():
    """
    Reset the global counters a rebuilt model's serialisation depends on.

    ``namer`` is the third of them: a declarative factor's ``name`` -- which the
    graphical pass records as :class:`~autofit.graph_spec.FactorInfo` and which
    ``graph.info`` prints -- comes from it, so a model *rebuilt* without
    resetting it is genuinely a model whose factors are named differently (this
    is why ``test_autofit/graphical/info/conftest.py`` resets it too).
    """
    af.ModelObject._ids = itertools.count()
    af.Prior._ids = itertools.count()
    namer.reset()


class Wrapper:
    def __init__(self, centre=(0.0, 0.0), pixels=10, normalization=1.0):
        self.centre = centre
        self.pixels = pixels
        self.normalization = normalization


def _rich_model():
    """
    One model exercising every extraction path: nesting, sharing, a relation, an
    assertion, tuple priors, a fixed tuple constant, a ``Model(int)`` and an
    instance leaf.
    """
    first = af.Model(af.ex.Gaussian)
    second = af.Model(af.ex.Gaussian)
    second.centre = first.centre
    second.sigma = first.normalization * 2.0

    nfw = af.Model(PhysicalNFW)
    wrapper = af.Model(
        Wrapper,
        centre=(0.1, 0.2),
        pixels=af.Model(int),
        normalization=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
    )

    model = af.Collection(
        first=first,
        second=second,
        nfw=nfw,
        wrapper=wrapper,
        instance=af.ex.Gaussian(centre=1.0, normalization=2.0, sigma=3.0),
    )
    model.add_assertion(first.sigma > 0.5)
    return model


def test_extracting_twice_is_byte_identical():
    model = _rich_model()

    first = json.dumps(GraphSpec.from_model(model).to_dict())
    second = json.dumps(GraphSpec.from_model(model).to_dict())

    assert first == second


def test_rebuilding_the_model_is_byte_identical():
    first = json.dumps(GraphSpec.from_model(_rich_model()).to_dict())

    _reset_ids()
    second = json.dumps(GraphSpec.from_model(_rich_model()).to_dict())

    assert first == second


def test_factor_graph_extraction_is_deterministic():
    def build():
        model_1 = af.Model(af.ex.Gaussian)
        model_2 = model_1.copy()
        model_2.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
        return af.FactorGraphModel(
            af.AnalysisFactor(prior_model=model_1, analysis=NullAnalysis()),
            af.AnalysisFactor(prior_model=model_2, analysis=NullAnalysis()),
        ).global_prior_model

    first = json.dumps(GraphSpec.from_model(build()).to_dict())
    _reset_ids()
    second = json.dumps(GraphSpec.from_model(build()).to_dict())

    assert first == second


def test_fixed_tuple_constant_and_model_int_are_never_dropped():
    """
    Both vanished from the prototype's figure, and their absence reads as
    absence from the model.  They are rows.
    """
    spec = GraphSpec.from_model(
        af.Model(
            Wrapper,
            centre=(0.1, 0.2),
            pixels=af.Model(int),
            normalization=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        )
    )

    centre = spec.row(("centre",))
    assert centre is not None
    assert centre.dimensionality == "tuple"
    assert centre.sampling == "fixed"
    assert [component.value for component in centre.components] == [0.1, 0.2]
    assert [component.name for component in centre.components] == [
        "centre_0",
        "centre_1",
    ]
    # `model.info` prints the whole tuple on one line, so both slots resolve to
    # the grouped parent path.
    assert spec.path_index["centre/centre_0"] == (("centre",),)

    pixels = spec.row(("pixels",))
    assert pixels is not None
    # Rule R7: `Model(int)` is a row on its owner, never a component.
    assert spec.node(("pixels",)) is None
    assert pixels.prior_cls_name == "int"
    # A zero-parameter `Model(int)` prints nothing in `model.info`, so the row
    # is an explicit added annotation.
    assert pixels.in_model_info is False
    assert spec.path_index["pixels"] == ()

    # Two fixed tuple slots + the `Model(int)` slot.
    assert spec.counts["fixed_leaf_slots"] == 3
    assert spec.counts["unique_sampled_scalars"] == 1


def test_missing_configuration_is_its_own_state():
    class Unconfigured:
        def __init__(self, areas_factor=1.0):
            self.areas_factor = areas_factor

    spec = GraphSpec.from_model(af.Model(Unconfigured))
    row = spec.row(("areas_factor",))

    assert row.sampling == "missing"
    assert row.prior_cls_name == "ConfigException"
    # A missing value *is* in `model.info` -- it is never silently absent.
    assert row.in_model_info is True
    assert spec.path_index["areas_factor"] == (("areas_factor",),)
    assert spec.counts["missing"] == 1


def test_solved_paths_are_marked_absent_from_model_info():
    spec = GraphSpec.from_model(
        af.Collection(gaussian=af.Model(af.ex.Gaussian)),
        solved_paths=("gaussian.normalization",),
    )
    row = spec.row(("gaussian", "normalization"))

    assert row.sampling == "solved"
    assert row.in_model_info is False
    assert spec.path_index["gaussian/normalization"] == ()


def test_collapse_is_on_by_default_and_can_be_turned_off():
    """
    Phase 1 accepted ``collapse`` and ignored it; the collapse phase fills it in.
    ``collapse=False`` must still return the *uncollapsed* tree unchanged -- it
    is the tree every catalogue construct is asserted against.
    """
    model = af.Collection(
        a=af.Model(af.ex.Gaussian),
        b=af.Model(af.ex.Gaussian),
    )
    collapsed = GraphSpec.from_model(model, collapse=True)
    expanded = GraphSpec.from_model(model, collapse=False)

    assert json.dumps(collapsed.to_dict()) != json.dumps(expanded.to_dict())
    assert json.dumps(GraphSpec.from_model(model).to_dict()) == json.dumps(
        collapsed.to_dict()
    )

    assert all(node.plate is None for node in expanded.components())
    assert expanded.counts["components"] == expanded.counts["components_raw"] == 3
    assert expanded.counts["plates"] == 0

    assert [node.plate.count for node in collapsed.components() if node.plate] == [2]
    assert collapsed.counts["components"] == 2
    assert collapsed.counts["components_raw"] == 3
    assert collapsed.counts["plates"] == 1


def test_importing_graph_spec_does_not_import_matplotlib():
    """
    Layer 1 knows nothing about pixels: importing it must not drag a drawing
    library in.  Checked in a *fresh* interpreter, since the test session has
    already imported matplotlib.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import autofit.graph_spec; "
            "print([m for m in sys.modules if m.split('.')[0] == 'matplotlib'])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"
