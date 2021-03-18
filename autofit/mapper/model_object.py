import itertools
from collections import Iterable


class ModelObject:
    _ids = itertools.count()

    def __init__(self):
        self.id = next(self._ids)

    @property
    def component_number(self):
        return self.id

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        try:
            return self.id == other.id
        except AttributeError:
            return False

    @property
    def identifier(self):
        hash_list = list()

        def _add_value_to_hash_list(value):
            if hasattr(value, "__dict__"):
                for key, value in value.__dict__.items():
                    if not (key.startswith("_") or key == "id"):
                        hash_list.append(key)
                        add_value_to_hash_list(
                            value
                        )
            elif isinstance(value, Iterable):
                for value in value:
                    add_value_to_hash_list(
                        value
                    )
            else:
                try:
                    h = hash(value)

                    hash_list.append(
                        h
                    )
                except ValueError:
                    pass

        def add_value_to_hash_list(value):
            if hasattr(value, "identifier"):
                hash_list.append(
                    value.identifier
                )
            else:
                _add_value_to_hash_list(
                    value
                )

        _add_value_to_hash_list(self)
        return hash(tuple(hash_list))
