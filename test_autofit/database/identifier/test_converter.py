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
    print(search.paths.output_path)
