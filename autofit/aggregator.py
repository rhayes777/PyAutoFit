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
from collections import defaultdict
from typing import List

import autofit.optimize.non_linear.non_linear
from autofit.optimize.non_linear.output import AbstractOutput


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
    def output(self) -> AbstractOutput:
        """
        An object describing the output data from the nonlinear search performed in this phase
        """
        return AbstractOutput(model=self.model, paths=self.optimizer.paths)

    @property
    def model_results(self) -> str:
        """
        Reads the model.results file
        """
        with open(os.path.join(self.directory, "model.results")) as f:
            return f.read()

    @property
    def dataset(self):
        """
        The dataset that this phase ran on
        """
        with open(
            os.path.join(self.directory, f"{self.dataset_name}.pickle"), "rb"
        ) as f:
            return pickle.load(f)

    @property
    def header(self) -> str:
        """
        A header created by joining the pipeline, phase and dataset names
        """
        return "/".join((self.pipeline, self.phase, self.dataset_name))

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


class AggregatorGroup:
    def __init__(self, groups: ["AbstractAggregator"]):
        """
        A group of aggregators produced by grouping phases on a field.

        Parameters
        ----------
        groups
            Groups, each with a common value in the metadata file
        """
        self.groups = groups

    def filter(self, **kwargs) -> "AggregatorGroup":
        """
        Apply a filter to the underlying groups whilst maintaining the total number of groups.

        Parameters
        ----------
        kwargs
            Key word arguments for filtering

        Returns
        -------
        A collection of groups of the same length with each group having the same or fewer members.
        """
        return AggregatorGroup([group.filter(**kwargs) for group in self.groups])

    def __getitem__(self, item):
        return self.groups[item]

    def __len__(self):
        return len(self.groups)

    def __getattr__(self, item):
        return [getattr(group, item) for group in self.groups]


class AbstractAggregator:
    def __init__(self, phases: List[PhaseOutput]):
        """
        An aggregator that comprises several phases which matching filters.

        Parameters
        ----------
        phases
            Phases that were found to have matching filters
        """
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

    def filter(self, **kwargs) -> "AbstractAggregator":
        """
        Filter by key value pairs found in the metadata.

        Another aggregator object is returned.

        Parameters
        ----------
        kwargs
            Key value arguments which are expected to match those found the in metadata file.

        Returns
        -------
        An aggregator comprising all phases that match the filters.
        """
        return AbstractAggregator(phases=self.phases_with(**kwargs))

    def __getattr__(self, item):
        """
        If an attribute is not found then attempt to grab a list of values from the underlying phases
        """
        return [getattr(phase, item) for phase in self.phases]

    def group_by(self, field: str) -> AggregatorGroup:
        """
        Group the phases by a field, e.g. pipeline.

        The object returned still permits filtering and attribute querying.

        Parameters
        ----------
        field
            The field by which to group

        Returns
        -------
        An object comprising lists of grouped fields
        """
        group_dict = defaultdict(list)
        for phase in self.phases:
            group_dict[getattr(phase, field)].append(phase)
        return AggregatorGroup(list(map(AbstractAggregator, group_dict.values())))

    @property
    def model_results(self) -> str:
        """
        A string joining headers and results for all included phases.
        """
        return "\n\n".join(
            "{}\n\n{}".format(phase.header, phase.model_results)
            for phase in self.phases
        )


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
            print(f"\nNo phases found in {directory}\n")
        else:
            paths_string = "\n".join(phase.directory for phase in phases)
            print(f"\nPhases were found in these directories:\n\n{paths_string}\n")
        super().__init__(phases)


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
        out.write(Aggregator(root_directory).filter(**filter_dict).model_results)
