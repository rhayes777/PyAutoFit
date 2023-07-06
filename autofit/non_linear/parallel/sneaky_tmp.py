import cProfile
import logging
import multiprocessing as mp
import os
from os import path
from typing import Iterable, Optional, Callable

from dynesty.dynesty import _function_wrapper
from emcee.ensemble import _FunctionWrapper
from mpi4py import MPI, MPIPoolExecutor

from autoconf import conf
from autofit.non_linear.paths.abstract import AbstractPaths
from .process import AbstractJob, Process, StopCommand

logger = logging.getLogger(
    __name__
)

class FunctionCache:
    """
    Singleton class to cache the functions and optional arguments between calls
    """


def initializer(
        fitness,
        prior_transform,
        fitness_args,
        fitness_kwargs,
        prior_transform_args,
        prior_transform_kwargs
):
    """
    Initialized function used to initialize the
    singleton object inside each worker of the pool
    """
    FunctionCache.fitness = fitness
    FunctionCache.prior_transform = prior_transform
    FunctionCache.fitness_args = fitness_args
    FunctionCache.fitness_kwargs = fitness_kwargs
    FunctionCache.prior_transform_args = prior_transform_args
    FunctionCache.prior_transform_kwargs = prior_transform_kwargs


def fitness_cache(x):
    """
    Likelihood function call
    """
    return FunctionCache.fitness(x, *FunctionCache.fitness_args,
                                 **FunctionCache.fitness_kwargs)


def prior_transform_cache(x):
    """
    Prior transform call
    """
    return FunctionCache.prior_transform(x, *FunctionCache.prior_transform_args,
                                         **FunctionCache.prior_transform_kwargs)

class SneakyPool:
    def __init__(
            self,
            processes: int,
            fitness: Callable,
            prior_transform: Optional[Callable] = None,
            fitness_args: Optional[Iterable] = None,
            fitness_kwargs: Optional[dict] = None,
            prior_transform_args: Optional[Iterable] = None,
            prior_transform_kwargs: Optional[dict] = None,
    ):
        
        self.fitness_init = fitness
        self.prior_transform_init = prior_transform
        self.fitness = fitness_cache
        self.prior_transform = prior_transform_cache
        self.fitness_args = fitness_args or ()
        self.fitness_kwargs = fitness_kwargs or {}
        self.prior_transform_args = prior_transform_args or ()
        self.prior_transform_kwargs = prior_transform_kwargs or {}
        self.processes = processes
        self.pool = None
        self.comm = MPI.COMM_WORLD

    def check_if_mpi(self):

        if self.comm.Get_size() > 1:
            return True
        else:
            return False

    def __enter__(self):
        """
        Activate the mp / mpi pool
        """
        
        init_args = (
            self.fitness_init, self.prior_transform_init,
            self.fitness_args, self.fitness_kwargs,
            self.prior_transform_args, self.prior_transform_kwargs
        )

        use_mpi = self.check_if_mpi()

        if use_mpi:
            self.pool = MPIPoolExecutor(
                max_workers=self.processes,
                initializer=initializer,
                initargs=init_args
            )
        else:
            self.pool = mp.Pool(
                processes=self.processes,
                initializer=initializer,
                initargs=init_args
            )
        
        initializer(*init_args)

        return self

    def map(
            self, function: Callable,
            iterable: Iterable
    ):
        """
        Map a function over an iterable using the map method
        of the initialized pool.

        Parameters
        ----------
        function
            A function to map
        iterable
            An iterable to map over

        """
        return self.pool.map(function, iterable)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.pool.terminate()
        except:  # noqa
            pass
        try:
            del (FunctionCache.fitness, FunctionCache.prior_transform,
                 FunctionCache.fitness_args, FunctionCache.fitness_kwargs,
                 FunctionCache.prior_transform_args, FunctionCache.prior_transform_kwargs)
        except:  # noqa
            pass
    
    @property
    def size(self):

        return self.njobs
    
    def close(self):
        self.pool.close()
    
    def join(self):
        self.pool.join()