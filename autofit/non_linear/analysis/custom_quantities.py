class CustomQuantities:
    def __init__(self):
        self.names = []
        self.values = []

        self._current_row = []

    def _position(self, name):
        return self.names.index(name)

    def add(self, **kwargs):
        for name in kwargs:
            if name not in self.names:
                self.names.append(name)

        self.values.append([kwargs[name] for name in self.names])
