import itertools


class ModelObject(object):
    _ids = itertools.count()

    def __init__(self):
        self.id = next(self._ids)

    def __hash__(self):
        return self.id
