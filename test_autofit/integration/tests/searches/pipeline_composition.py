from os import path
from autoconf import conf
import autofit as af
from test_autofit.integration.src.dataset import dataset as ds
from test_autofit.integration.src.phase import phase as ph
from test_autofit.integration.src.model import profiles
from test_autofit.integration.tests import runner

import numpy as np

test_type = "searches"
data_name = "gaussian_x1"


def make_pipeline_0():

    pipeline_name = "pipeline_0"

    phase = ph.Phase(
        name="phase",
        profiles=af.CollectionPriorModel(gaussian=profiles.Gaussian),
        search=af.DynestyStatic(),
    )

    return af.Pipeline(pipeline_name, phase)


def make_pipeline_1():

    pipeline_name = "pipeline_1"

    class GridPhase(af.as_grid_search(phase_class=ph.Phase, parallel=False)):
        @property
        def grid_priors(self):
            return [self.model.profiles.gaussian.one_tuple]

    phase = GridPhase(
        name="phase_grid",
        profiles=af.CollectionPriorModel(gaussian=af.last.model.profiles.gaussian),
        search=af.DynestyStatic(n_live_points=40, evidence_tolerance=5.0),
        number_of_steps=2,
    )

    return af.Pipeline(pipeline_name, phase)


integration_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "..", "..")

conf.instance.push(
        new_path=path.join(integration_path, "config"),  
        output_path=path.join(integration_path, "output", test_type)
)

dataset = ds.Dataset.from_fits(
    data_path=path.join(integration_path, "dataset", data_name, "data.fits"),
    noise_map_path=path.join(integration_path, "dataset", data_name, "noise_map.fits"),
)

pipeline_0 = make_pipeline_0()

pipeline_1 = make_pipeline_1()

pipeline = pipeline_0 + pipeline_1

pipeline.run(dataset=dataset)
