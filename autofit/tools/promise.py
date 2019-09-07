class PromiseResult:
    def __init__(self, phase, *result_path, assert_exists=True):
        self.phase = phase
        self.assert_exists = assert_exists
        self.result_path = result_path

    @property
    def variable(self):
        return Promise(
            self.phase,
            result_path=self.result_path,
            assert_exists=self.assert_exists
        )

    @property
    def constant(self):
        return Promise(
            self.phase,
            result_path=self.result_path,
            is_constant=True,
            assert_exists=self.assert_exists
        )

    def __getattr__(self, item):
        return PromiseResult(
            self.phase,
            *self.result_path,
            item,
            assert_exists=False
        )


class Promise:
    def __init__(
            self,
            phase,
            *path,
            result_path,
            is_constant=False,
            assert_exists=True
    ):
        self.phase = phase
        self.path = path
        self.is_constant = is_constant
        self.assert_exists = assert_exists
        self.result_path = result_path
        if assert_exists:
            phase.variable.object_for_path(path)

    def __call__(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        if item in ("phase", "path", "is_constant"):
            return super().__getattribute__(item)
        return Promise(
            self.phase,
            *self.path,
            item,
            result_path=self.result_path,
            is_constant=self.is_constant,
            assert_exists=self.assert_exists
        )

    def populate(self, results_collection):
        results = results_collection.from_phase(self.phase.phase_name)
        for item in self.result_path:
            results = getattr(results, item)
        model = results.constant if self.is_constant else results.variable
        return model.object_for_path(self.path)
