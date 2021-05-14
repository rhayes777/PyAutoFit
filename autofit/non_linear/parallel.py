import collections
import multiprocessing
from abc import ABC, abstractmethod
from itertools import count
from typing import Iterable

from dynesty.dynesty import _function_wrapper

from autofit.non_linear.abstract_search import NonLinearSearch
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
    def perform(self, *args):
        """
        Perform the task and return the result
        """


class Process(multiprocessing.Process):

    def __init__(
            self,
            name: str,
            job_queue: multiprocessing.Queue,
            initializer=None,
            initargs=None,
            job_args=tuple()
    ):
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

        self.initializer = initializer
        self.initargs = initargs

        self.job_args = job_args

    def run(self):
        """
        Run this process, completing each job in the job_queue and
        passing the result to the queue.
        """
        if self.initializer is not None:
            if self.initargs is None:
                return self.initializer()
            if isinstance(
                    self.initargs,
                    collections.Iterable
            ):
                initargs = tuple(self.initargs)
            else:
                initargs = (self.initargs,)

            self.initializer(
                *initargs
            )

        logger.debug("starting process {}".format(self.name))
        while True:
            if self.job_queue.empty():
                break
            else:
                job = self.job_queue.get()
                self.queue.put(
                    job.perform(
                        *self.job_args
                    )
                )
        logger.debug("terminating process {}".format(self.name))
        self.job_queue.close()

    @classmethod
    def run_jobs(
            cls,
            jobs: Iterable[AbstractJob],
            number_of_cores: int,
            initializer=None,
            initargs=None,
            job_args=tuple()
    ):
        """
        Run the collection of jobs across n - 1 other cores.

        Parameters
        ----------
        job_args
        initargs
        initializer
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
            cls(
                str(number),
                job_queue,
                initializer=initializer,
                initargs=initargs,
                job_args=job_args
            )
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


def _is_likelihood_function(
        function
) -> bool:
    """
    Is the function a callable used to evaluate likelihood?

    Naturally in Autofit this would be a child of the Fitness class.
    In Dynesty the likelihood function is wrapped in _function_wrapper
    and called 'loglikelihood'

    Parameters
    ----------
    function
        Some object

    Returns
    -------
    Is the object a log likelihood function?
    """
    return isinstance(
        function,
        NonLinearSearch.Fitness
    ) or (
                   isinstance(
                       function,
                       _function_wrapper
                   ) and function.name == 'loglikelihood'
           )


class SneakyJob(AbstractJob):
    def __init__(self, function, *args):
        """
        A job performed on a process.

        The log likelhood function is filtered from the args, but its index retained.
        This prevents large amounts of data comprised in an Analysis class from being
        copied over to processes multiple times.

        Parameters
        ----------
        function
            Some function to which a pool.map has been applied
        args
            The arguments to that function
        """
        super().__init__()
        self.function = function

        self.args = list()
        self.fitness_index = None

        for i, arg in enumerate(args):
            if _is_likelihood_function(
                    arg
            ):
                if self.fitness_index is not None:
                    raise AssertionError(
                        f"Two arguments of type NonLinear.Fitness passed to {function.__name__}"
                    )
                self.fitness_index = i
            else:
                self.args.append(arg)

    def perform(self, likelihood_function):
        """
        Computes the log likelihood. The likelihood function
        is passed from a copy associated with the current process

        Parameters
        ----------
        likelihood_function
            A likelihood function associated with the processes
            to avoid copying data for every single function call

        Returns
        -------
        The log likelihood
        """
        args = (
                self.args[:self.fitness_index]
                + [likelihood_function]
                + self.args[self.fitness_index:]
        )
        return self.function(
            args
        )


class SneakyPool:
    def __init__(
            self,
            processes: int,
            fitness: NonLinearSearch.Fitness,
            initializer=None,
            initargs=None
    ):
        """
        Implements the same interface as multiprocessing's pool,
        but associates the fitness object with each process to
        prevent data being copies to each process for every function
        call.

        Parameters
        ----------
        processes
            The number of cores to be used simultaneously.
        fitness
            A class comprising data and a model which can be used
            to evaluate the likelihood of a live point
        initializer
        initargs
        """
        self.processes = processes
        self.initializer = initializer
        self.initargs = initargs
        self.fitness = fitness

    def map(self, function, args_list):
        """
        Execute the function with the given arguments across all of the
        processes. The likelihood  argument is removed from each args in
        the args_list.

        Parameters
        ----------
        function
            Some function
        args_list
            An iterable of iterables of arguments passed to the function

        Yields
        ------
        Results from the function evaluation
        """
        jobs = [
            SneakyJob(
                function,
                *(
                    (args,) if not isinstance(
                        args,
                        Iterable
                    ) else tuple(args)
                )
            ) for args in args_list
        ]

        for result in Process.run_jobs(
                jobs,
                self.processes,
                initializer=self.initializer,
                initargs=self.initargs,
                job_args=(self.fitness,),
        ):
            yield result
