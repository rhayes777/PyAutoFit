import autofit as af
from autofit import Gaussian


def test_consistent_identifier():
    search = af.DynestyStatic(
        name="name"
    )
    search.paths.model = af.PriorModel(
        Gaussian
    )
    search.paths.save_all(
        search_config_dict=search.config_dict_search,
        info={},
        pickle_files=[]
    )
    search_output = af.SearchOutput(
        search.paths.output_path
    )
    assert search_output.model is not None
    assert search_output.search.paths.identifier == search.paths.identifier
    print(search.paths.output_path)
