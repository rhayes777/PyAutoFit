import autofit as af
from test_autofit.integration.src.phase import phase as ph
from test_autofit.integration.src.model import profiles
from test_autofit.integration.tests import runner

test_type = "searches"
data_name = "gaussian_x1"

phase = ph.Phase(
    name="phase",
    profiles=af.CollectionPriorModel(gaussian=profiles.Gaussian),
    search=af.PySwarmsGlobal,
)

if __name__ == "__main__":
    import sys

    runner.run(module=sys.modules[__name__])
