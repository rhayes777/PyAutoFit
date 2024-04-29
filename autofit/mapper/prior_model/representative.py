from collections import defaultdict
from typing import List


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
        path_value_tuples = _find_groups(path_value_tuples, position)
    return path_value_tuples


def integers_representative_key(integers: List[int]) -> str:
    """
    Given a list of integers, return a string that represents them in a concise way.

    e.g.
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] -> "1 - 10"
    [1, 2, 3, 5, 6, 7, 8, 9, 10] -> "1 - 3, 5 - 10"
    [1, 2, 3, 5, 6, 7, 8, 9, 10, 12] -> "1 - 3, 5 - 10, 12"

    Parameters
    ----------
    integers
        The list of integers to represent

    Returns
    -------
    A string representing the integers in a concise way
    """
    integers = sorted(integers)
    ranges = []
    start = integers[0]
    end = integers[0]

    for integer in integers[1:]:
        if integer == end + 1:
            end = integer
        else:
            ranges.append((start, end))
            start = integer
            end = integer

    ranges.append((start, end))

    return ", ".join(
        f"{start} - {end}" if start != end else str(start) for start, end in ranges
    )


def _find_groups(path_value_tuples, position):
    groups = defaultdict(list)
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
            representative_key = integers_representative_key(names)
        except ValueError:
            representative_key = (
                f"{min(names)} - {max(names)}" if len(set(names)) > 1 else names[0]
            )

        paths.append(((*before, representative_key, *after), value))
    return paths
