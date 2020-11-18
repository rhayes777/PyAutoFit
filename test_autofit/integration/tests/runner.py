import os
from os import path
from autoconf import conf
from test_autofit.integration.src.dataset import dataset as ds


def run(module):

    integration_path = path.join("{}".format(os.path.dirname(os.path.realpath(__file__))), "..")

    conf.instance.push(
        new_path=path.join(integration_path,"config"),
        output_path=path.join(integration_path, "output", module.test_type),
    )

    dataset = ds.Dataset.from_fits(
        data_path=path.join(integration_path, "dataset",module.data_name, "data.fits"),
        noise_map_path=path.join(integration_path,"dataset",module.data_name, "noise_map.fits"),
    )

    module.phase.run(dataset=dataset)
