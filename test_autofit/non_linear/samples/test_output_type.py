import autofit as af
from autoconf.conf import with_config
from autofit import SamplesNest, SearchOutput


@with_config(
    "general",
    "output",
    "samples_to_csv",
    value=True,
)
def test_search_output(output_directory):
    paths = af.DirectoryPaths(
        "name",
        path_prefix=output_directory,
    )
    samples = SamplesNest(
        model=af.Model(af.Gaussian),
        sample_list=[],
        samples_info=None,
    )
    paths.save_samples(samples=samples)

    search_output = SearchOutput(paths.output_path)

    assert isinstance(search_output.samples, SamplesNest)
