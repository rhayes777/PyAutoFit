import copy
from os import path
from typing import List, Tuple, Union

import numpy as np

from autofit import exc
from autofit.mapper import model_mapper as mm
from autofit.mapper.prior import prior as p
from autofit.non_linear.parallel import AbstractJob, Process, AbstractJobResult
from autofit.non_linear.result import Result








class JobResult(AbstractJobResult):
    def __init__(self, result, result_list_row, number):
        """
        The result of a job

        Parameters
        ----------
        result
            The result of a grid search
        result_list_row
            A row in the result list
        """
        super().__init__(number)
        self.result = result
        self.result_list_row = result_list_row


class Job(AbstractJob):
    def __init__(self, search_instance, model, analysis, arguments, index):
        """
        A job to be performed in parallel.

        Parameters
        ----------
        search_instance
            An instance of an optimiser
        analysis
            An analysis
        arguments
            The grid search arguments
        """
        super().__init__()
        self.search_instance = search_instance
        self.analysis = analysis
        self.model = model
        self.arguments = arguments
        self.index = index

    def perform(self):
        result = self.search_instance.fit(model=self.model, analysis=self.analysis)
        result_list_row = [
            self.index,
            *[prior.lower_limit for prior in self.arguments.values()],
            result.log_likelihood,
        ]

        return JobResult(result, result_list_row, self.number)



