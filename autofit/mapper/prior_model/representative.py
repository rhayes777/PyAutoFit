from collections import defaultdict
from typing import List, Tuple

from .collection import Collection
from ..prior.abstract import Prior


def value_hash(value):
    if isinstance(value, Prior):
        return hash(
            tuple(
                getattr(value, arg) for arg in value.__database_args__ if arg != "id_"
            )
        )
    return hash(value)


def find_groups(path_value_tuples):
    groups = defaultdict(list)
    for path, value in path_value_tuples:
        root_name, *rest = path
        groups[(tuple(rest), value_hash(value))].append(root_name)
    return groups


class Representative(Collection):
    def __init__(self, collection):
        super().__init__(collection.items())

    @property
    def info_tuples(self) -> List[Tuple]:
        info_tuples = list(super().info_tuples)
        return info_tuples
