import logging

logger = logging.getLogger(__name__)


class ResultsCollection(list):
    def __init__(self, results=None):
        super().__init__(results or [])

    @property
    def last(self):
        if len(self) > 0:
            return self[-1]
        return None

    @property
    def first(self):
        if len(self) > 0:
            return self[0]
        return None


class Pipeline(object):

    def __init__(self, pipeline_name, *phases):
        """

        Parameters
        ----------
        pipeline_name: str
            The phase_name of this pipeline
        """
        self.pipeline_name = pipeline_name
        self.phases = phases

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
        return self.__class__("{} + {}".format(self.pipeline_name, other.pipeline_name), *(self.phases + other.phases))

    def run_function(self, func):
        results = ResultsCollection()
        for i, phase in enumerate(self.phases):
            logger.info("Running Phase {} (Number {})".format(phase.optimizer.name, i))
            results.append(func(phase, results.copy()))
        return results
