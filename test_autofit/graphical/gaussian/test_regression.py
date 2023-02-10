import dill

import autofit as af
import autofit.graphical as ep
from test_autofit.graphical.gaussian.model import Gaussian, Analysis


def test_pickle():
    prior_model = af.Model(
        Gaussian
    )
    analysis_factor = ep.AnalysisFactor(
        prior_model,
        analysis=Analysis(
            x=1,
            y=2
        ),
    )

    analysis_factor = dill.loads(
        dill.dumps(analysis_factor)
    )

    assert isinstance(
        analysis_factor,
        ep.AnalysisFactor
    )
