import multiprocessing as mp
from typing import Iterable

from dynesty.dynesty import _function_wrapper
from emcee.ensemble import _FunctionWrapper

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
    from autofit.non_linear.abstract_search import NonLinearSearch
    return any([
        isinstance(
            function,
            NonLinearSearch.Fitness
        ),
        isinstance(
            function,
            _function_wrapper
        ) and function.name == 'loglikelihood',
        isinstance(
            function,
            _FunctionWrapper
        )
    ])


class SneakyJob(AbstractJob):
    def __init__(self, function, *args):
        """
        A job performed on a process.

        If the function is the log likelihood function then it is set to None.

        If the log likelihood function is in the args, it is filtered from the args,
        but its index retained.

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
        if _is_likelihood_function(function):
            self.function = None
        else:
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
        is passed from a copy associated with the current process.

        Depending on whether the likelihood function itself is
        being mapped, or some function mapped onto the likelihood
        function as an argument, the likelihood function will be
        called or added to the arguments.

        Parameters
        ----------
        likelihood_function
            A likelihood function associated with the processes
            to avoid copying data for every single function call

        Returns
        -------
        The log likelihood
        """
        if self.function is None:
            return likelihood_function(
                self.args
            )
        args = (
                self.args[:self.fitness_index]
                + [likelihood_function]
                + self.args[self.fitness_index:]
        )
        return self.function(
            args
        )


class StopCommand:
    """
    A command that can be passed into a process queue to gracefully stop
    the process
    """


class SneakyProcess(Process):
    def run(self):
        """
        Run this process, completing each job in the job_queue and
        passing the result to the queue.

        The process continues to execute until a StopCommand is passed.
        This occurs when the SneakyMap goes out of scope.
        """

        self._init()
        self.logger.debug("starting")
        while True:
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
            fitness,
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

        exception = None

        while count < target:
            for process in self.processes:
                if not process.queue.empty():
                    item = process.queue.get()
                    count += 1
                    if isinstance(
                            item,
                            Exception
                    ):
                        exception = item
                    else:
                        yield item

        if exception is not None:
            raise exception

    def __del__(self):
        """
        Called when the map goes out of scope.

        Tell each process to terminate with a StopCommand and then join
        each process with a timeout of one second.
        """
        for _ in range(len(self.processes)):
            self.job_queue.put(StopCommand)

        for process in self.processes:
            process.join(0.5)
