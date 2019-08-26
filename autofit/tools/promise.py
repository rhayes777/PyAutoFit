class Promise:
    def __init__(
            self,
            phase,
            *path,
            is_constant=False
    ):
        self.phase = phase
        self.path = path
        self.is_constant = is_constant

    def __getattr__(self, item):
        return Promise(
            self.phase,
            *self.path,
            item,
            is_constant=self.is_constant
        )
