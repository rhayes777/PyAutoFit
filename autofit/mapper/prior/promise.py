from abc import ABC, abstractmethod

from autofit.mapper.prior.arithmetic import ArithmeticMixin
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

    def model_absolute(self, a) -> "AbstractPromise":
        """
        Parameters
        ----------
        a
            The absolute width of gaussian priors

        Returns
        -------
        A promise for a prior or model with absolute gaussian widths
        """
        model = self.model
        model.absolute = a
        return model

    def model_relative(self, r) -> "AbstractPromise":
        """
        Parameters
        ----------
        r
            The relative width of gaussian priors

        Returns
        -------
        A promise for a prior or model with relative gaussian widths
        """
        model = self.model
        model.relative = r
        return model

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

    def __init__(self, *result_path, index=0):
        super().__init__(*result_path)
        self._index = index

    def __getitem__(self, item: int) -> "LastPromiseResult":
        """
        LastPromiseResults can be indexed.

        An index of 0 gives a LPR corresponding to the latest Results.
        An index of -1 gives a LPR corresponding the the Results before last.
        etc.
        """
        if item > 0:
            raise IndexError("last only accepts negative indices")
        return LastPromiseResult(*self.result_path, index=item)

    @property
    def model(self):
        """
        A promise for an object in the model result. This might be a prior or prior model.
        """
        return LastPromise(result_path=self.result_path, index=self._index)

    @property
    def instance(self):
        """
        A promise for an object in the best fit result. This must be an instance or instance.
        """
        return LastPromise(
            result_path=self.result_path, is_instance=True, index=self._index
        )

    def __getattr__(self, item):
        if item == "_index":
            return super().__getattr__(item)
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
        self._phase = phase
        self.assert_exists = assert_exists

    def __getstate__(self):
        return {}

    @property
    def model(self):
        """
        A promise for an object in the model result. This might be a prior or prior model.
        """
        return Promise(
            self._phase,
            result_path=self.result_path,
            assert_exists=self.assert_exists
        )

    @property
    def instance(self):
        """
        A promise for an object in the best fit result. This must be an instance or instance.
        """
        return Promise(
            self._phase,
            result_path=self.result_path,
            is_instance=True,
            assert_exists=self.assert_exists,
        )

    def __getattr__(self, item):
        return PromiseResult(
            self._phase,
            *self.result_path,
            item,
            assert_exists=False
        )


class AbstractPromise(ABC, ArithmeticMixin):
    def __init__(
            self,
            *path,
            result_path,
            is_instance=False,
            absolute=None,
            relative=None,
            is_optional=False,
    ):
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
            True if the promised object belongs to the instance result object. Otherwise it belongs to the
            model.
        """
        self.path = path
        self.is_instance = is_instance
        self.result_path = result_path
        self.absolute = absolute
        self.relative = relative
        self.is_optional = is_optional

    @property
    def attribute_name(self) -> str:
        """
        The name of the attribute which was called to create this prior.
        """
        return "instance" if self.is_instance else "model"

    @property
    def path_string(self):
        """
        A string expressing the path of attributes called to create this prior.
        """
        return ".".join(map(str, self.path))

    def __str__(self):
        return f"result.{self.attribute_name}.{self.path_string}"

    @property
    def settings(self):
        return dict(
            is_instance=self.is_instance,
            absolute=self.absolute,
            relative=self.relative,
            is_optional=self.is_optional,
        )

    def __call__(self, *args, **kwargs):
        pass

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
            try:
                results = getattr(results, item)
            except AttributeError as e:
                if self.is_optional:
                    return None
                raise e
        if self.absolute is not None:
            model = results.model_absolute(self.absolute)
        elif self.relative is not None:
            model = results.model_relative(self.relative)
        elif self.is_instance:
            model = results.instance
        else:
            model = results.model

        try:
            return model.object_for_path(self.path)
        except AttributeError as e:
            if self.is_optional:
                return None
            raise e


class Promise(AbstractPromise):
    def __init__(
            self,
            phase,
            *path,
            result_path,
            is_instance=False,
            assert_exists=True,
            relative=None,
            absolute=None,
            is_optional=False,
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
        super().__init__(
            *path,
            result_path=result_path,
            is_instance=is_instance,
            absolute=absolute,
            relative=relative,
            is_optional=is_optional,
        )
        self._phase = phase
        self.assert_exists = assert_exists
        if assert_exists:
            phase.model.object_for_path(path)

    def __str__(self):
        return f"{self._phase.phase_name}.{super().__str__()}"

    def __getstate__(self):
        return {}

    def __getattr__(self, item):
        if item in (
                "_phase",
                "path",
                "is_instance",
                "_populate_from_results",
                "optional",
                "is_optional",
        ):
            return super().__getattribute__(item)
        return Promise(
            self._phase,
            *self.path,
            item,
            result_path=self.result_path,
            assert_exists=self.assert_exists,
            **self.settings,
        )

    def __getitem__(self, item):
        return Promise(
            self._phase,
            *self.path,
            item,
            result_path=self.result_path,
            assert_exists=self.assert_exists,
            **self.settings,
        )

    @property
    def optional(self) -> "Promise":
        """
        Make this promise evaluate to None if a corresponding float or prior
        does not exist at evaluation time. This is instead of raising an
        AttributeError.
        """
        return Promise(
            self._phase,
            *self.path,
            result_path=self.result_path,
            assert_exists=False,
            **{
                key: value
                for key, value in self.settings.items()
                if key != "is_optional"
            },
            is_optional=True,
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
        results = results_collection.from_phase(
            self._phase.phase_name
        )
        return self._populate_from_results(results)


class LastPromise(AbstractPromise):
    """
    A promise that searches the results collection to find the latest result that contains an object with the
    specified path
    """

    def __init__(
            self,
            *path,
            result_path,
            is_instance=False,
            index=0,
            absolute=None,
            relative=None,
            is_optional=False,
    ):
        self._index = index
        super().__init__(
            *path,
            result_path=result_path,
            is_instance=is_instance,
            relative=relative,
            absolute=absolute,
            is_optional=is_optional,
        )

    def __getattr__(self, item):
        if item in (
                "phase",
                "path",
                "is_instance",
                "_populate_from_results",
                "optional",
                "is_optional",
        ):
            return super().__getattribute__(item)
        return LastPromise(
            *self.path,
            item,
            result_path=self.result_path,
            index=self._index,
            **self.settings,
        )

    @property
    def optional(self):
        return LastPromise(
            *self.path,
            result_path=self.result_path,
            index=self._index,
            **{
                key: value
                for key, value in self.settings.items()
                if key != "is_optional"
            },
            is_optional=True,
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
        for results in list(results_collection.reversed)[-self._index:]:
            try:
                return self._populate_from_results(results)
            except AttributeError:
                pass
        raise AttributeError(
            f"No attribute found with path {self.path} in previous phase"
        )


last = LastPromiseResult()
