#!/usr/bin/env python

"""
Filter and collate phase simulator in all subdirectories.

Usage:

./aggregator.py (root_directory) [pipeline=pipeline phase=phase simulator=simulator]

Example:

./aggregator.py ../output pipeline=data_mass_x1_source_x1_positions
"""

import os
import pickle
import zipfile
from typing import List

import autofit.optimize.non_linear.non_linear
from autofit.optimize.non_linear.output import Output


class PhaseOutput:
    """
    @DynamicAttrs
    """

    def __init__(self, directory: str):
        """
        Represents the output of a single phase. Comprises a metadata file and other simulator files.

        Parameters
        ----------
        directory
            The directory of the phase
        """
        self.directory = directory
        self.__optimizer = None
        self.__model = None
        self.file_path = os.path.join(directory, "metadata")
        with open(self.file_path) as f:
            self.text = f.read()
            pairs = [line.split("=") for line in self.text.split("\n")]
            self.__dict__.update({pair[0]: pair[1] for pair in pairs})

    @property
    def output(self) -> Output:
        """
        An object describing the output data from the nonlinear search performed in this phase
        """
        return Output(
            model=self.model,
            paths=self.optimizer.paths
        )

    @property
    def model_results(self) -> str:
        """
        Reads the model.results file
        """
        with open(os.path.join(self.directory, "model.results")) as f:
            return f.read()

    @property
    def header(self) -> str:
        """
        A header created by joining the pipeline, phase and dataset names
        """
        return "/".join((self.pipeline, self.phase, self.dataset))

    @property
    def optimizer(self) -> autofit.optimize.non_linear.non_linear.NonLinearOptimizer:
        """
        The optimizer object that was used in this phase
        """
        if self.__optimizer is None:
            with open(os.path.join(self.directory, "optimizer.pickle"), "r+b") as f:
                self.__optimizer = pickle.loads(f.read())
        return self.__optimizer

    @property
    def model(self) -> autofit.optimize.non_linear.non_linear.NonLinearOptimizer:
        """
        The optimizer object that was used in this phase
        """
        if self.__model is None:
            with open(os.path.join(self.directory, "model.pickle"), "r+b") as f:
                self.__model = pickle.loads(f.read())
        return self.__model

    def __str__(self):
        return self.text

    def __repr__(self):
        return "<PhaseOutput {}>".format(self)


class AbstractAggregator:
    def __init__(self, phases: List[PhaseOutput]):
        self.phases = phases

    def __getitem__(self, item):
        return self.phases[item]

    def __len__(self):
        return len(self.phases)

    def phases_with(self, **kwargs) -> [PhaseOutput]:
        """
        Filters phases. If no arguments are passed all phases are returned. Arguments must be key value pairs, with
        phase, simulator or pipeline as the key.

        Parameters
        ----------
        kwargs
            Filters, e.g. pipeline=pipeline1
        """
        return [
            phase
            for phase in self.phases
            if all([getattr(phase, key) == value for key, value in kwargs.items()])
        ]

    def filter(self, **kwargs):
        return AbstractAggregator(
            phases=self.phases_with(
                **kwargs
            )
        )

    def __getattr__(self, item):
        return [
            getattr(phase, item)
            for phase in self.phases
        ]


class Aggregator(AbstractAggregator):
    def __init__(self, directory: str):
        """
        Class to aggregate phase results for all subdirectories in a given directory.

        The whole directory structure is traversed and a Phase object created for each directory that contains a
        metadata file.

        Parameters
        ----------
        directory
        """
        self.directory = directory
        phases = []

        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(".zip"):
                    with zipfile.ZipFile(f"{root}/{filename}", "r") as f:
                        f.extractall(root)

        for root, _, filenames in os.walk(directory):
            if "metadata" in filenames:
                phases.append(PhaseOutput(root))

        if len(phases) == 0:
            print(
                f"\nNo phases found in {directory}\n"
            )
        else:
            paths_string = "\n".join(
                phase.directory for phase in phases
            )
            print(
                f"\nPhases were found in these directories:\n\n{paths_string}\n"
            )
        super().__init__(phases)

    def optimizers_with(
            self, **kwargs
    ) -> [autofit.optimize.non_linear.non_linear.NonLinearOptimizer]:
        """
        Load a list of optimizers for phases in the directory with zero or more filters applied.

        Parameters
        ----------
        kwargs
            Filters, e.g. pipeline=pipeline1

        Returns
        -------
        optimizers
            A list of optimizers, one for each phase in the directory that matches the filters.
        """
        return [phase.optimizer for phase in self.phases_with(**kwargs)]

    def outputs_with(
            self, **kwargs
    ) -> [Output]:
        """
        Load a list of optimizer outputs for phases in the directory with zero or more filters applied.

        Parameters
        ----------
        kwargs
            Filters, e.g. pipeline=pipeline1

        Returns
        -------
        outputs
            A list of outputs, one for each phase
        """
        return [phase.output for phase in self.phases_with(**kwargs)]

    def model_results(self, **kwargs) -> str:
        """
        Collates model results from all phases in the directory or some subset if filters are applied.

        Parameters
        ----------
        kwargs
            Filters, e.g. pipeline=pipeline1

        Returns
        -------
        model_results
            A string joining headers and results for all included phases.
        """
        return "\n\n".join(
            "{}\n\n{}".format(phase.header, phase.model_results)
            for phase in self.phases_with(**kwargs)
        )


if __name__ == "__main__":
    from sys import argv

    root_directory = None
    try:
        root_directory = argv[1]
    except IndexError:
        print(
            "Usage:\n\naggregator.py (root_directory) [pipeline=pipeline phase=phase simulator=simulator]"
        )
        exit(1)
    filter_dict = {pair[0]: pair[1] for pair in [line.split("=") for line in argv[2:]]}

    with open("model.results", "w+") as out:
        out.write(Aggregator(root_directory).model_results(**filter_dict))
