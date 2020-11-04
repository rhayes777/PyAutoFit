import multiprocessing
from abc import ABC, abstractmethod
from time import sleep
from typing import List

from autofit.non_linear.log import logger


class AbstractJob(ABC):
    @abstractmethod
    def perform(self):
        pass


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
    def run_jobs(cls, jobs: List[AbstractJob], number_of_cores):
        job_queue = multiprocessing.Queue()

        processes = [
            Process(str(number), job_queue)
            for number in range(number_of_cores - 1)
        ]

        for job in jobs:
            job_queue.put(job)

        for process in processes:
            process.start()

        count = 0

        while count < len(jobs):
            for process in processes:
                while not process.queue.empty():
                    result = process.queue.get()
                    count += 1
                    yield result

        job_queue.close()

        for process in processes:
            process.join(timeout=1.0)
