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


def find_groups(path_value_tuples, limit=0):
    maximum_path_length = max(len(path) for path, _ in path_value_tuples)
    for position in range(maximum_path_length - limit):
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
            groups[(tuple(before), tuple(after), value_hash)].append((root_name, value))
        except IndexError:
            paths.append((path, value))

    for (before, after, _), name_values in groups.items():
        names, values = zip(*name_values)
        representative_key = (
            f"{min(names)} - {max(names)}" if len(set(names)) > 1 else names[0]
        )
        paths.append(((*before, representative_key, *after), values[0]))
    return paths
