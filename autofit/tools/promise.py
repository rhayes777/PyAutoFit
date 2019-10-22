class PromiseResult:
    def __init__(self, phase, *result_path, assert_exists=True):
        """
        A wrapper for a phase that facilitates the generation of priors by consuming user defined paths.

        Parameters
        ----------
        phase
            A phase from which a promise is generated
        result_path
            The path through the results objects to the correct result. This is used for phase extensions when
            a result object can have child result objects
        assert_exists
            If this is true then an AttributeError is thrown if there is no object with the given path in the
            model of the origin phase
        """
        self.phase = phase
        self.assert_exists = assert_exists
        self.result_path = result_path

    @property
    def variable(self):
        """
        A promise for an object in the variable result. This might be a prior or prior model.
        """
        return Promise(
            self.phase, result_path=self.result_path, assert_exists=self.assert_exists
        )

    @property
    def constant(self):
        """
        A promise for an object in the best fit result. This must be an instance or constant.
        """
        return Promise(
            self.phase,
            result_path=self.result_path,
            is_constant=True,
            assert_exists=self.assert_exists,
        )

    def __getattr__(self, item):
        return PromiseResult(self.phase, *self.result_path, item, assert_exists=False)


class Promise:
    def __init__(
        self, phase, *path, result_path, is_constant=False, assert_exists=True
    ):
        """
        Place holder for an object in the object hierarchy. This is replaced at runtime by a prior, prior
        model, constant or instance

        Parameters
        ----------
        phase
            The phase that holds or creates the promised object
        path
            The path to the promised object. e.g. if a phase has a variable galaxies.lens.phi then the path
            will be ("galaxies", "lens", "phi")
        result_path
            The path through the result collection to the result object required. This is used for hyper phases
            where a result object can have child result objects for each phase extension.
        is_constant
            True if the promised object belongs to the constant result object
        assert_exists
            If this is true then an exception is raised if an object is not defined in the addressed phase's
            model. Hyper phases are a bit trickier so no assertion is made.
        """
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
        """
        Find the object that this promise corresponds to be getting results for a particular phase and then
        traversing those results using the path.

        Parameters
        ----------
        results_collection
            A collection of results from previous phases

        Returns
        -------
        obj
            The promised prior, prior model, instance or constant
        """
        results = results_collection.from_phase(self.phase.phase_name)
        for item in self.result_path:
            results = getattr(results, item)
        model = results.constant if self.is_constant else results.variable
        return model.object_for_path(self.path)
