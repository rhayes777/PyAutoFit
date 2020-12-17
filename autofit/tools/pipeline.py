import logging
import copy
from autofit import exc

logger = logging.getLogger(__name__)


class ResultsCollection:
    def __init__(self):
        """
        A collection of results from previous phases. Results can be obtained using an index or the name of the phase
        from whence they came.
        """
        self.__result_list = []
        self.__result_dict = {}

    def copy(self):
        collection = ResultsCollection()
        collection.__result_dict = self.__result_dict
        collection.__result_list = self.__result_list
        return collection

    @property
    def reversed(self):
        return reversed(self.__result_list)

    @property
    def last(self):
        """
        The result of the last phase
        """
        if len(self.__result_list) > 0:
            return self.__result_list[-1]
        return None

    @property
    def first(self):
        """
        The result of the first phase
        """
        if len(self.__result_list) > 0:
            return self.__result_list[0]
        return None

    def add(self, name, result):
        """
        Add the result of a phase.

        Parameters
        ----------
        name: str
            The name of the phase
        result
            The result of that phase
        """
        try:
            self.__result_list[self.__result_list.index(result)] = result
        except ValueError:
            self.__result_list.append(result)
        self.__result_dict[name] = result

    def __getitem__(self, item):
        """
        Get the result of a previous phase by index

        Parameters
        ----------
        item: int
            The index of the result

        Returns
        -------
        result: Result
            The result of a previous phase
        """
        return self.__result_list[item]

    def __len__(self):
        return len(self.__result_list)

    def from_phase(self, name):
        """
        Returns the result of a previous phase by its name

        Parameters
        ----------
        name: str
            The name of a previous phase

        Returns
        -------
        result: Result
            The result of that phase

        Raises
        ------
        exc.PipelineException
            If no phase with the expected result is found
        """
        try:
            return self.__result_dict[name]
        except KeyError:
            raise exc.PipelineException(
                "No previous phase named {} found in results ({})".format(
                    name, ", ".join(self.__result_dict.keys())
                )
            )

    def __contains__(self, item):
        return item in self.__result_dict


class Pipeline:
    def __init__(self, pipeline_name, path_prefix, results, *phases):
        """
        A pipeline of phases to be run sequentially. Results are passed between phases. Phases must have unique names.

        Parameters
        ----------
        pipeline_name: str
            The name of this pipeline
        """
        self.pipeline_name = pipeline_name
        self.path_prefix = path_prefix
        self.results = copy.deepcopy(results)
        self.phases = phases
        self.pipeline_tag = None

        for phase in phases:

            if path_prefix is not None:
                phase.search.paths.path_prefix = path_prefix

            if phase.pipeline_name is None:
                phase.pipeline_name = pipeline_name
            if phase.pipeline_tag is None:
                phase.pipeline_tag = self.pipeline_tag

        phase_names = [phase.name for phase in phases]

        if len(set(phase_names)) < len(phase_names):
            raise exc.PipelineException(
                "Cannot create pipelines with duplicate phase names. ({})".format(
                    ", ".join(phase_names)
                )
            )

    def __getitem__(self, item):
        return self.phases[item]

    def __add__(self, other):
        """
        Compose two runners

        Parameters
        ----------
        other: Pipeline
            Another pipeline

        Returns
        -------
        composed_pipeline: Pipeline
            A pipeline that runs all the  phases from this pipeline and then all the phases from the other pipeline
        """
        return self.__class__(
            "{} + {}".format(
                self.pipeline_name,
                other.pipeline_name
            ),
            None,
            other.results,
            *(self.phases + other.phases),
        )

    def run(self, dataset):
        def runner(phase, results):
            return phase.run(dataset=dataset, results=results)

        return self.run_function(runner)

    def run_function(self, func):
        """
        Run the function for each phase in the pipeline.

        Parameters
        ----------
        func
            A function that takes a phase and prior results, returning results for that phase

        Returns
        -------
        results: ResultsCollection
            A collection of results
        """
        if self.results is None:
            results = ResultsCollection()
        else:
            results = self.results

        for i, phase in enumerate(self.phases):
            logger.info(
                "Running Phase {} (Number {})".format(
                    phase.name,
                    i
                )
            )
            name = phase.name
            results.add(name, func(phase, results))
        return results
