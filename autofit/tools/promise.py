from abc import ABC, abstractmethod

from autofit.tools.pipeline import ResultsCollection


class AbstractPromiseResult(ABC):
    def __init__(self, *result_path):
        self.result_path = result_path

    @property
    @abstractmethod
    def model(self):
        """
        A promise for an object in the model result. This might be a prior or prior model.
        """

    @property
    @abstractmethod
    def instance(self):
        """
        A promise for an object in the best fit result. This must be an instance or instance.
        """

    @abstractmethod
    def __getattr__(self, item):
        """
        Used to manage results paths
        """

    def __call__(self, *args, **kwargs):
        pass


class LastPromiseResult(AbstractPromiseResult):
    """
    A PromiseResult that does not require a phase. Refers to the latest Result in the collection with an object for
    the path specified in a promise.
    """

    @property
    def model(self):
        """
        A promise for an object in the model result. This might be a prior or prior model.
        """
        return LastPromise(result_path=self.result_path)

    @property
    def instance(self):
        """
        A promise for an object in the best fit result. This must be an instance or instance.
        """
        return LastPromise(result_path=self.result_path, is_instance=True)

    def __getattr__(self, item):
        return LastPromiseResult(*self.result_path, item)


class PromiseResult(AbstractPromiseResult):
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
        super().__init__(*result_path)
        self.phase = phase
        self.assert_exists = assert_exists

    @property
    def model(self):
        """
        A promise for an object in the model result. This might be a prior or prior model.
        """
        return Promise(
            self.phase, result_path=self.result_path, assert_exists=self.assert_exists
        )

    @property
    def instance(self):
        """
        A promise for an object in the best fit result. This must be an instance or instance.
        """
        return Promise(
            self.phase,
            result_path=self.result_path,
            is_instance=True,
            assert_exists=self.assert_exists,
        )

    def __getattr__(self, item):
        return PromiseResult(self.phase, *self.result_path, item, assert_exists=False)


class AbstractPromise(ABC):
    def __init__(self, *path, result_path, is_instance=False):
        """
        Place holder for an object in the object hierarchy. This is replaced at runtime by a prior, prior
        model, instance or instance

        Parameters
        ----------
        path
            The path to the promised object. e.g. if a phase has a model galaxies.lens.phi then the path
            will be ("galaxies", "lens", "phi")
        result_path
            The path through the result collection to the result object required. This is used for hyper phases
            where a result object can have child result objects for each phase extension.
        is_instance
            True if the promised object belongs to the instance result object
        """
        self.path = path
        self.is_instance = is_instance
        self.result_path = result_path

    def __call__(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        if item in ("phase", "path", "is_instance"):
            return super().__getattribute__(item)
        return Promise(
            *self.path, item, result_path=self.result_path, is_instance=self.is_instance
        )

    @abstractmethod
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
            The promised prior, prior model, instance or instance
        """

    def _populate_from_results(self, results):
        for item in self.result_path:
            results = getattr(results, item)
        model = results.instance if self.is_instance else results.model
        return model.object_for_path(self.path)


class Promise(AbstractPromise):
    def __init__(
        self, phase, *path, result_path, is_instance=False, assert_exists=True
    ):
        """
        Place holder for an object in the object hierarchy. This is replaced at runtime by a prior, prior
        model, instance or instance

        Parameters
        ----------
        phase
            The phase that holds or creates the promised object
        path
            The path to the promised object. e.g. if a phase has a model galaxies.lens.phi then the path
            will be ("galaxies", "lens", "phi")
        result_path
            The path through the result collection to the result object required. This is used for hyper phases
            where a result object can have child result objects for each phase extension.
        is_instance
            True if the promised object belongs to the instance result object
        assert_exists
            If this is true then an exception is raised if an object is not defined in the addressed phase's
            model. Hyper phases are a bit trickier so no assertion is made.
        """
        super().__init__(*path, result_path=result_path, is_instance=is_instance)
        self.phase = phase
        self.assert_exists = assert_exists
        if assert_exists:
            phase.model.object_for_path(path)

    def __getattr__(self, item):
        if item in ("phase", "path", "is_instance", "_populate_from_results"):
            return super().__getattribute__(item)
        return Promise(
            self.phase,
            *self.path,
            item,
            result_path=self.result_path,
            is_instance=self.is_instance,
            assert_exists=self.assert_exists,
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
            The promised prior, prior model, instance or instance
        """
        results = results_collection.from_phase(self.phase.phase_name)
        return self._populate_from_results(results)


class LastPromise(AbstractPromise):
    """
    A promise that searches the results collection to find the latest result that contains an object with the
    specified path
    """

    def __getattr__(self, item):
        if item in ("phase", "path", "is_instance", "_populate_from_results"):
            return super().__getattribute__(item)
        return LastPromise(
            *self.path, item, result_path=self.result_path, is_instance=self.is_instance
        )

    def populate(self, results_collection: ResultsCollection):
        """
        Recover the instance or model associated with this promise from the latest result in the results collection
        where a matching path is found.
        
        Parameters
        ----------
        results_collection
            A collection of results to be searched in reverse

        Returns
        -------
        A prior, model or instance.
        
        Raises
        ------
        AttributeError
            If no matching prior is found
        """
        for results in results_collection.reversed:
            try:
                return self._populate_from_results(results)
            except AttributeError:
                pass
        raise AttributeError(
            f"No attribute found with path {self.path} in previous phase"
        )


last = LastPromiseResult()
