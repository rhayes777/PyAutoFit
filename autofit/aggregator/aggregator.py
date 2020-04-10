#!/usr/bin/env python

"""
Filter and collate phase dataset in all subdirectories.

Usage:

./aggregator.py (root_directory) [pipeline=pipeline phase=phase dataset=dataset]

Example:

./aggregator.py ../output pipeline=data_mass_x1_source_x1_positions
"""

import os
import zipfile
from collections import defaultdict
from shutil import rmtree
from typing import List, Union, Iterator

from .phase_output import PhaseOutput
from .predicate import AttributePredicate


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

    def filter(self, *predicates) -> "AggregatorGroup":
        """
        Apply predicates to the underlying groups whilst maintaining the total number of groups.

        Parameters
        ----------
        predicates
            Predicates that evaluate to True or False for any given phase.

        Returns
        -------
        A collection of groups of the same length with each group having the same or fewer members.
        """
        return AggregatorGroup(
            [
                group.filter(*predicates)
                for group in self.groups
            ]
        )

    def __getitem__(self, item):
        return self.groups[item]

    def __len__(self):
        return len(self.groups)

    def values(self, name: str) -> List[List]:
        """
        Extract a list of lists values with a certain name from the output objects in
        this group.

        Parameters
        ----------
        name
            The name of the attribute to be extracted

        Returns
        -------
        A list of lists of values.
        """
        return [
            group.values(name)
            for group in self.groups
        ]


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

    def remove_unzipped(self):
        """
        Removes the unzipped output directory for each phase.
        """
        for phase in self.phases:
            path = "/".join(
                phase.directory.split("/")[:-1]
            )
            rmtree(
                path,
                ignore_errors=True
            )

    def __getitem__(
            self,
            item: Union[slice, int]
    ) -> Union[
        "AbstractAggregator",
        PhaseOutput
    ]:
        """
        If an index is passed in then a specific phase output is returned.

        If a slice is passed in then an aggregator comprising several phases is returned.

        Parameters
        ----------
        item
            A slice or index

        Returns
        -------
        An aggregator or phase
        """
        if isinstance(item, slice):
            return AbstractAggregator(
                self.phases[item]
            )
        return self.phases[item]

    def __len__(self):
        return len(self.phases)

    def __getattr__(self, item):
        return AttributePredicate(item)

    def filter(self, *predicates) -> "AbstractAggregator":
        """
        Filter phase outputs by predicates. A predicate is created using a conditional
        operator.

        Another aggregator object is returned.

        Parameters
        ----------
        predicates
            Objects representing predicates that may evaluate to True or False for any
            given phase output.

        Returns
        -------
        An aggregator comprising all phases that evaluate to True for all predicates.
        """
        phases = self.phases
        for predicate in predicates:
            phases = predicate.filter(phases)
        return AbstractAggregator(phases=list(phases))

    def values(self, name: str) -> Iterator:
        """
        Get values from outputs with a given name.

        A list the same length as the number of phases is returned
        where each item is the value of the attribute for a given
        phase.

        Parameters
        ----------
        name
            The name of an attribute expected to be associated with
            phase output. If a pickle file with this name is in the
            phase output directory then that pickle will be loaded.

        Returns
        -------
        A generator of values for the attribute
        """
        return map(
            lambda phase: getattr(
                phase, name
            ),
            self.phases
        )

    def map(self, func):
        """
        Map some function onto the aggregated output objects.

        Parameters
        ----------
        func
            A function

        Returns
        -------
        A generator of results
        """
        return map(
            func,
            self.phases
        )

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
    def __init__(
            self,
            directory: str,
            completed_only=False
    ):
        """
        Class to aggregate phase results for all subdirectories in a given directory.

        The whole directory structure is traversed and a Phase object created for each directory that contains a
        metadata file.

        Parameters
        ----------
        directory
            A directory in which the outputs of phases are kept. This is searched recursively.
        completed_only
            If True only phases with a .completed file (indicating the phase was completed)
            are included in the aggregator.
        """
        self._directory = directory
        phases = []

        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(".zip"):
                    with zipfile.ZipFile(f"{root}/{filename}", "r") as f:
                        f.extractall(f"{root}/{filename[:-4]}")

        for root, _, filenames in os.walk(directory):
            if "metadata" in filenames:
                if not completed_only or ".completed" in filenames:
                    phases.append(PhaseOutput(root))

        if len(phases) == 0:
            print(f"\nNo phases found in {directory}\n")
        else:
            paths_string = "\n".join(phase.directory for phase in phases)
            print(f"\nPhases were found in these directories:\n\n{paths_string}\n")
        super().__init__(phases)
