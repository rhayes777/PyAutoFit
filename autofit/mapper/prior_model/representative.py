from collections import defaultdict
<<<<<<< HEAD
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
=======


def find_groups(path_value_tuples, limit=1):
    """
    Groups path-value tuples by their path, up to a limit. This is used to group priors in the model representation.

    If multiple paths share the same suffix they are grouped together.

    Parameters
    ----------
    path_value_tuples
        A list of tuples of paths and values in the model
    limit
        How far from the end of paths grouping should terminate. This can be used to prevent grouping of
        priors and attributes.

    Returns
    -------
    A list of tuples of paths and values, where paths are grouped by their suffix.
    """
    try:
        maximum_path_length = max(len(path) for path, _ in path_value_tuples)
    except ValueError:
        return path_value_tuples
    for position in range(maximum_path_length - limit):
>>>>>>> feature/docs
        path_value_tuples = _find_groups(path_value_tuples, position)
    return path_value_tuples


def _find_groups(path_value_tuples, position):
    groups = defaultdict(list)
<<<<<<< HEAD
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


class Representative(Collection):
    def __init__(self, collection):
        super().__init__(collection.items())

    @property
    def info_tuples(self) -> List[Tuple]:
        info_tuples = list(super().info_tuples)
        return info_tuples
=======
    paths = []
    for path, value in path_value_tuples:
        try:
            root_name = path[position]
            before = path[:position]
            after = path[position + 1 :]
            groups[(tuple(before), tuple(after), value)].append(root_name)
        except IndexError:
            paths.append((path, value))

    for (before, after, value), names in groups.items():
        try:
            names = list(map(int, names))
        except ValueError:
            pass

        representative_key = (
            f"{min(names)} - {max(names)}" if len(set(names)) > 1 else names[0]
        )
        paths.append(((*before, representative_key, *after), value))
    return paths
>>>>>>> feature/docs
