from typing import Dict, Union, Tuple

from autofit.mapper.prior_model.collection import Collection
from autofit.mapper.prior.gaussian import GaussianPrior


def simple_model_for_kwargs(kwargs: Dict[Union[str, Tuple[str, ...]], float]):
    model = Collection()
    for path, value in kwargs.items():
        if isinstance(path, str):
            path = (path,)
        component = model
        if len(path) > 1:
            for part in path[:1]:
                try:
                    component = component[part]
                except KeyError:
                    new_component = Collection()
                    component[part] = new_component
                    component = new_component

        component[path[-1]] = GaussianPrior(
            mean=value,
            sigma=0.0,
        )

    return model
