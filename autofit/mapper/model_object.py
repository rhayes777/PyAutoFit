import itertools


class ModelObject(object):
    _ids = itertools.count()

    def __init__(self):
        self.id = next(self._ids)

    @property
    def component_number(self):
        return self.id

    def __hash__(self):
        return self.id

    def __gt__(self, other):
        return self.id > other.id

    def __eq__(self, other):
        try:
            return self.id == other.id
        except AttributeError:
            return False
