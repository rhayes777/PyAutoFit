import pickle

import autofit as af


def test_serialize_optimiser():
    optimizer = af.DynestyStatic(
        "name"
    )

    optimizer = pickle.loads(
        pickle.dumps(
            optimizer
        )
    )
    assert optimizer.name == "name"
