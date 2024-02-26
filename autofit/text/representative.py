from collections import defaultdict
from typing import Dict

import autofit as af


class Representative:
    def __init__(self, items):
        self.items = items

    @property
    def keys(self):
        return [key for key, _ in self.items]

    @property
    def children(self):
        return [obj for _, obj in self.items]

    @property
    def key(self):
        keys = sorted(self.keys)
        return f"{keys[0]} - {keys[-1]}"

    @property
    def representative(self):
        return self.items[0][1]

    def __getattr__(self, item):
        return getattr(self.representative, item)

    def __len__(self):
        return len(self.representative)

    @classmethod
    def find_representatives(cls, items, minimum=4):
        representative_dict: Dict[tuple, list] = defaultdict(list)
        for key, obj in items:
            blueprint = cls.get_blueprint(obj)
            representative_dict[blueprint].append((key, obj))

        representatives = []
        for blueprint, items in representative_dict.items():
            if len(items) >= minimum:
                representative = Representative(items)
                representatives.append((representative.key, representative))
            else:
                representatives.extend(items)

        return representatives

    @classmethod
    def get_blueprint(cls, obj):
        from autofit.text.formatter import FormatNode

        if obj is None:
            return None

        if isinstance(obj, FormatNode):
            return cls.get_blueprint(obj.value)
        if isinstance(obj, (float, int, tuple, str)):
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
