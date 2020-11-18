import multiprocessing
from abc import ABC, abstractmethod
from itertools import count
from time import sleep
from typing import Iterable

from autofit.non_linear.log import logger


class AbstractJobResult(ABC):
    def __init__(self, number):
        self.number = number

    def __eq__(self, other):
        return self.number == other.number

    def __gt__(self, other):
        return self.number > other.number

    def __lt__(self, other):
        return self.number < other.number


class AbstractJob(ABC):
    """
    Task to be completed in parallel
    """
    _number = count()

    def __init__(self):
        self.number = next(self._number)

    @abstractmethod
    def perform(self):
        """
        Perform the task and return the result
        """


class Process(multiprocessing.Process):
    def __init__(self, name: str, job_queue: multiprocessing.Queue):
        """
        A parallel process that consumes Jobs through the job queue and outputs results through its own queue.

        Parameters
        ----------
        name: str
            The name of the process
        job_queue: multiprocessing.Queue
            The queue through which jobs are submitted
        """
        super().__init__(name=name)
        logger.info("created process {}".format(name))

        self.job_queue = job_queue
        self.queue = multiprocessing.Queue()
        self.count = 0
        self.max_count = 250

    def run(self):
        """
        Run this process, completing each job in the job_queue and
        passing the result to the queue.
        """
        logger.info("starting process {}".format(self.name))
        while True:
            sleep(0.025)
            if self.count >= self.max_count:
                break
            if self.job_queue.empty():
                self.count += 1
            else:
                self.count = 0
                job = self.job_queue.get()
                self.queue.put(job.perform())
        logger.info("terminating process {}".format(self.name))
        self.job_queue.close()

    @classmethod
    def run_jobs(
            cls,
            jobs: Iterable[AbstractJob],
            number_of_cores: int
    ):
        """
        Run the collection of jobs across n - 1 other cores.

        Parameters
        ----------
        jobs
            Serializable concrete children of the AbstractJob class
        number_of_cores
            The number of cores this computer has. Must be at least 2.
        """
        if number_of_cores < 2:
            raise AssertionError(
                "The number of cores available must be at least 2 for parallel to run"
            )

        job_queue = multiprocessing.Queue()

        processes = [
            Process(str(number), job_queue)
            for number in range(number_of_cores - 1)
        ]

        total = 0
        for job in jobs:
            job_queue.put(job)
            total += 1

        for process in processes:
            process.start()

        count = 0

        while count < total:
            for process in processes:
                while not process.queue.empty():
                    result = process.queue.get()
                    count += 1
                    yield result

        job_queue.close()

        for process in processes:
            process.join(timeout=1.0)
