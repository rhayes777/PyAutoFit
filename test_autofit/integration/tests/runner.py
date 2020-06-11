import os
from autoconf import conf
from test_autofit.integration.src.dataset import dataset as ds

import numpy as np

def run(module):

    integration_path = "{}/..".format(os.path.dirname(os.path.realpath(__file__)))

    conf.instance = conf.Config(
        config_path=f"{integration_path}/config",
        output_path=f"{integration_path}/output/{module.test_type}"
    )

    dataset = ds.Dataset.from_fits(
        data_path=f"{integration_path}/dataset/{module.data_name}/data.fits",
        noise_map_path=f"{integration_path}/dataset/{module.data_name}/noise_map.fits",
    )

    mask = np.full(fill_value=False, shape=dataset.data.shape)

    module.phase.run(dataset=dataset, mask=mask)