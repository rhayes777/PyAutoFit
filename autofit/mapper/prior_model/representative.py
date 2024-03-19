from collections import defaultdict

from ..prior.abstract import Prior


def _value_hash(value):
    if isinstance(value, Prior):
        return hash(
            tuple(
                getattr(value, arg) for arg in value.__database_args__ if arg != "id_"
            )
        )
    return hash(value)


def find_groups(path_value_tuples):
    maximum_path_length = max(len(path) for path, _ in path_value_tuples)
    for position in range(maximum_path_length - 1):
        path_value_tuples = _find_groups(path_value_tuples, position)
    return path_value_tuples


def _find_groups(path_value_tuples, position):
    groups = defaultdict(list)
    value_map = {}
    paths = []
    for path, value in path_value_tuples:
        try:
            root_name = path[position]
            before = path[:position]
            after = path[position + 1 :]
            value_hash = _value_hash(value)
            if value_hash not in value_map:
                value_map[value_hash] = value
            groups[(tuple(before), tuple(after), value_hash)].append(root_name)
        except IndexError:
            paths.append((path, value))
    return paths + [
        (
            (
                *before,
                f"{min(names)} - {max(names)}" if len(names) > 1 else names[0],
                *after,
            ),
            value_map[value_hash],
        )
        for (before, after, value_hash), names in groups.items()
    ]
