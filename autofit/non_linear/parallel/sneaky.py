import multiprocessing as mp
from time import sleep
from typing import Iterable

from dynesty.dynesty import _function_wrapper

from autofit.non_linear.abstract_search import NonLinearSearch
from .process import AbstractJob, Process


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


class StopException(Exception):
    pass


class StopCommand:
    pass


class SneakyProcess(Process):
    def run(self):
        """
        Run this process, completing each job in the job_queue and
        passing the result to the queue.
        """

        self._init()
        self.logger.debug("starting")
        while True:
            if self.job_queue.empty():
                sleep(1)
            else:
                job = self.job_queue.get()
                if job is StopCommand:
                    break
                try:
                    self.queue.put(
                        job.perform(
                            *self.job_args
                        )
                    )
                except Exception as e:
                    self.queue.put(e)
        self.logger.debug("terminating process {}".format(self.name))
        self.job_queue.close()


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
        self.job_queue = mp.Queue()
        self.processes = [
            SneakyProcess(
                str(number),
                self.job_queue,
                initializer=initializer,
                initargs=initargs,
                job_args=(fitness,)
            )
            for number in range(processes)
        ]
        for process in self.processes:
            process.start()

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

        for job in jobs:
            self.job_queue.put(
                job
            )

        target = len(jobs)
        count = 0

        while count < target:
            for process in self.processes:
                if not process.queue.empty():
                    item = process.queue.get()
                    if isinstance(
                            item,
                            Exception
                    ):
                        raise item
                    count += 1
                    yield item

    def __del__(self):
        print("del")
        for _ in range(len(self.processes)):
            self.job_queue.put(StopCommand)

        for process in self.processes:
            process.join(1)
        print("done")
