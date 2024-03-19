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
    maximum_path_length = max(len(path) for path, _ in path_value_tuples)
    for position in range(maximum_path_length):
        path_value_tuples = _find_groups(path_value_tuples, position)
    return path_value_tuples


def _find_groups(path_value_tuples, position):
    groups = defaultdict(list)
    for path, value in path_value_tuples:
        root_name = path[position]
        before = path[:position]
        after = path[position + 1 :]
        groups[(tuple(before), tuple(after), value_hash(value))].append(root_name)
    return [
        (
            (
                *before,
                f"{min(names)} - {max(names)}" if len(names) > 1 else names[0],
                *after,
            ),
            value,
        )
        for (before, after, value), names in groups.items()
    ]
