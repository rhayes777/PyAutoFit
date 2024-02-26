from collections import defaultdict
from typing import Dict

import autofit as af


class Representative:
    def __init__(self, children):
        self.children = children

    @classmethod
    def find_representatives(cls, collection):
        representative_dict: Dict[tuple, list] = defaultdict(list)
        for model in collection:
            blueprint = cls.get_blueprint(model)
            representative_dict[blueprint].append(model)

        representatives = []
        for blueprint, models in representative_dict.items():
            if len(models) > 1:
                representatives.append(Representative(models))
            else:
                representatives.extend(models)

        return representatives

    @classmethod
    def get_blueprint(cls, obj):
        if isinstance(obj, (float, int, tuple)):
            return obj
        if isinstance(obj, af.Prior):
            return type(obj), obj.parameter_string
        if isinstance(obj, af.AbstractModel):
            return tuple(
                (path, cls.get_blueprint(value))
                for path, value in obj.path_instance_tuples_for_class(
                    (float, int, tuple, af.Prior), ignore_children=True
                )
                if path != ("id",)
            )
        raise ValueError(f"Cannot get blueprint for {obj} of type {type(obj)}")
