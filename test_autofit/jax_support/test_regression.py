from autofit.database import Object
from autofit.jax_wrapper import numpy as np


def test_model():
    assert Object.from_object(np.array(1.0))() == 1.0
