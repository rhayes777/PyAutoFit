import os
import autoarray as aa
import autofit as af
from test_autofit.integration import integration_util
from autofit.non_linear.mock.mock_nlo import MockNLO


def run(module, test_name=None, non_linear_class=af.MultiNest, config_folder="config"):
    test_name = test_name or module.test_name
    test_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))
    output_path = test_path + "output/"
    config_path = test_path + config_folder
    conf.instance = conf.Config(config_path=config_path, output_path=output_path)
    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=test_path, folder_names=["dataset", module.data_type]
    )

    imaging = aa.Imaging.from_fits(
        image_path=f"{dataset_path}/image.fits",
        noise_map_path=f"{dataset_path}/noise_map.fits",
        pixel_scales=0.1,
    )

    module.make_pipeline_no_lens_light(
        name=test_name,
        phase_folders=[module.test_type, test_name],
        non_linear_class=non_linear_class,
    ).run(dataset=imaging)


def run_a_mock(module):
    # noinspection PyTypeChecker
    run(
        module,
        test_name=f"{module.test_name}_mock",
        non_linear_class=MockNLO,
        config_folder="config_mock",
    )


def run_with_multi_nest(module):
    # noinspection PyTypeChecker
    run(
        module,
        test_name=f"{module.test_name}_nest",
        non_linear_class=af.MultiNest,
        config_folder="config_mock",
    )
