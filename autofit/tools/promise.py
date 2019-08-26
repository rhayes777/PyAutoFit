class Promise:
    def __init__(
            self,
            phase,
            *path
    ):
        self.phase = phase
        self.path = path

    def __getattr__(self, item):
        return Promise(
            self.path,
            *self.path,
            item
        )
