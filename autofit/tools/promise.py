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
        phase.variable.object_for_path(path)

    def __getattr__(self, item):
        if item in ("phase", "path", "is_constant"):
            return super().__getattribute__(item)
        return Promise(
            self.phase,
            *self.path,
            item,
            is_constant=self.is_constant
        )

    def populate(self, results_collection):
        results = results_collection.from_phase(self.phase.phase_name)
        model = results.constant if self.is_constant else results.variable
        return model.object_for_path(self.path)
