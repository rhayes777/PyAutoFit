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
            fitness,
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
