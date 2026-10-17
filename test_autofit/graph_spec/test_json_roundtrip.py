import json

import autofit as af
from autofit.graph_spec import GraphSpec
from autofit.mapper.identifier import Identifier


class Parameterless:
    def __init__(self):
        pass


def semantic_graph(value):
    # GraphSpec exposes process-local object/prior counters. JSON has never
    # preserved these; compare every semantic field, separately pin folder IDs.
    if isinstance(value, dict):
        return {
            k: semantic_graph(v)
            for k, v in value.items()
            if k not in {"obj_id", "prior_id"}
        }
    if isinstance(value, list):
        return [semantic_graph(v) for v in value]
    return value


def test_fixed_models_survive_json(tmp_path):
    model = af.Collection(
        fixed=af.Model(af.ex.Gaussian, centre=0.0, normalization=0.1, sigma=0.01),
        empty=af.Model(Parameterless),
        free=af.Model(af.ex.Gaussian),
    )
    original_id = str(Identifier(model))
    original_graph = GraphSpec.from_model(model).to_dict()
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model.dict()))
    reloaded = af.Model.from_json(path)
    assert isinstance(reloaded.fixed, af.Model)
    assert isinstance(reloaded.empty, af.Model)
    assert semantic_graph(GraphSpec.from_model(reloaded).to_dict()) == semantic_graph(
        original_graph
    )
    assert str(Identifier(model)) == original_id
    assert str(Identifier(reloaded)) == original_id
    before = af.Nautilus(name="roundtrip")
    before.paths.model = model
    after = af.Nautilus(name="roundtrip")
    after.paths.model = reloaded
    assert after.paths.identifier == before.paths.identifier
    assert after.paths.output_path == before.paths.output_path


def test_fixed_model_identifier_compatibility():
    model = af.Model(af.ex.Gaussian, centre=0.0, normalization=0.1, sigma=0.01)
    assert model.dict()["type"] == "model"
    # Pinned from unmodified main, before the serialization repair.
    assert str(Identifier(model)) == "47764ddfa16f140e6596b381544285c4"
