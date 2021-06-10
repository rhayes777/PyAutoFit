import logging
import math
from itertools import count
from multiprocessing import Process
from multiprocessing import Queue

from autofit.non_linear.parallel.sneaky import StopCommand

logger = logging.getLogger(
    __name__
)


class AnalysisProcess(Process):
    _id = count()

    def __init__(
            self,
            analyses,
            analysis_queues
    ):
        super().__init__()
        self.analyses = analyses
        self.analysis_queues = analysis_queues
        self.queue = Queue()

        self.name = f"analysis_process_{next(self._id)}"

        self.logger = logging.getLogger(
            self.name
        )

    def _run(self):
        while True:
            for analysis, queue in zip(
                    self.analyses,
                    self.analysis_queues
            ):
                instance = queue.get()

                if instance is StopCommand:
                    return

                try:
                    self.queue.put(
                        analysis.log_likelihood_function(
                            instance
                        )
                    )
                except Exception as e:
                    self.queue.put(e)

    def run(self):
        """
        Run this process, completing each job in the job_queue and
        passing the result to the queue.
        """

        self.logger.debug("starting")

        self._run()

        self.logger.debug("terminating")
        self.queue.close()


class AnalysisPool:
    def __init__(
            self,
            analyses,
            n_cores
    ):
        self.analyses = analyses
        self.n_cores = n_cores

        self.n_analyses = len(analyses)
        n_processes = min(
            self.n_analyses,
            n_cores
        )

        analyses_per_process = math.ceil(
            self.n_analyses / n_processes
        )

        self.processes = list()
        self.analysis_queues = list()

        for n in range(
                n_processes
        ):
            analyses_ = analyses[
                        n * analyses_per_process: (n + 1) * analyses_per_process
                        ]
            analysis_queues_ = [
                Queue()
                for _
                in analyses_
            ]
            process = AnalysisProcess(
                analyses=analyses_,
                analysis_queues=analysis_queues_
            )
            self.analysis_queues.extend(analysis_queues_)

            self.processes.append(
                process
            )
            process.start()

    def __del__(self):
        """
        Called when the map goes out of scope.

        Tell each process to terminate with a StopCommand and then join
        each process with a timeout of one second.
        """
        logger.debug(
            "Deconstructing SneakyMap"
        )

        logger.debug(
            "Terminating processes..."
        )
        for analysis_queue in self.analysis_queues:
            analysis_queue.put(StopCommand)

        logger.debug(
            "Joining processes..."
        )
        for process in self.processes:
            try:
                process.join(0.5)
            except Exception as e:
                logger.exception(e)

    def __call__(self, instance):
        log_likelihood = 0

        count_ = 0

        for queue in self.analysis_queues:
            queue.put(instance)

        while count_ < self.n_analyses:
            for process in self.processes:
                result = process.queue.get()

                if isinstance(
                        result,
                        Exception
                ):
                    raise result

                log_likelihood += result
                count_ += 1

        return log_likelihood
