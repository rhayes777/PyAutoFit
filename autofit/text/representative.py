from collections import defaultdict
from typing import Dict, List, Iterable

import autofit as af


class Representative:
    def __init__(self, items: List[tuple]):
        """
        Collects together items that are the same except for ids. This is to make
        the output of the info method more concise.

        Parameters
        ----------
        items
            A list of tuples of the form (key, object)
        """
        self.items = items

    @property
    def keys(self):
        return [key for key, _ in self.items]

    @property
    def children(self):
        return [obj for _, obj in self.items]

    @property
    def key(self) -> str:
        """
        A string representation of the range of keys in this representative.
        """
        keys = sorted(self.keys)
        return f"{keys[0]} - {keys[-1]}"

    @property
    def representative(self):
        """
        The first object in the group represents the group.
        """
        return self.items[0][1]

    def __getattr__(self, item):
        return getattr(self.representative, item)

    def __len__(self):
        return len(self.representative)

    @classmethod
    def find_representatives(cls, items: Iterable[tuple], minimum: int = 4) -> list:
        """
        Find representatives in a list of items. This includes items from
        the original list where there are not enough repetitions to form
        a representative.

        Parameters
        ----------
        items
            A list of tuples of the form (key, object)
        minimum
            The minimum number of items that must be the same for a representative
            to be formed.

        Returns
        -------
        A list of representatives and items that are not part of a representative.
        """
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
        """
        Get a blueprint for an object. This is a tuple of tuples of the form
        (path, value) where path is a tuple of strings and value is a float, int,
        tuple, or af.Prior.

        Blueprints are unique per unique object in the model but are not sensitive
        to ids.

        Parameters
        ----------
        obj
            The object to get a blueprint for.

        Returns
        -------
        A blueprint for the object.
        """
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
            blueprint = tuple(
                (path, cls.get_blueprint(value))
                for path, value in obj.path_instance_tuples_for_class(
                    (float, int, tuple, af.Prior), ignore_children=True
                )
                if path != ("id",)
            )
            if isinstance(obj, af.Model):
                return blueprint + (obj.cls,)
            return blueprint
        raise ValueError(f"Cannot get blueprint for {obj} of type {type(obj)}")
