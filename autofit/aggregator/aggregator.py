#!/usr/bin/env python

"""
Filter and collate search outputs in all subdirectories.

Usage:

./aggregator.py (root_directory) [pipeline=pipeline phase=phase dataset=dataset]

Example:

./aggregator.py ../output pipeline=data_mass_x1_source_x1_positions
"""

import os
import zipfile
from collections import defaultdict
from os import path
from shutil import rmtree
from typing import List, Union, Iterator

from .predicate import AttributePredicate
from .search_output import SearchOutput


class AggregatorGroup:
    def __init__(self, groups: ["AbstractAggregator"]):
        """
        A group of aggregators produced by grouping search_outputs on a field.

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
            Predicates that evaluate to `True` or `False` for any given phase.

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
    def __init__(self, search_outputs: List[SearchOutput]):
        """
        An aggregator that comprises several search_outputs which matching filters.

        Parameters
        ----------
        search_outputs
            search_outputs that were found to have matching filters
        """
        self.search_outputs = search_outputs

    def remove_unzipped(self):
        """
        Removes the unzipped output directory for each phase.
        """
        for phase in self.search_outputs:

            split_path = path.split(phase.directory)[0]

            unzipped_path = path.join(split_path)

            rmtree(
                unzipped_path,
                ignore_errors=True
            )

    def __getitem__(
            self,
            item: Union[slice, int]
    ) -> Union[
        "AbstractAggregator",
        SearchOutput
    ]:
        """
        If an index is passed in then a specific phase output is returned.

        If a slice is passed in then an aggregator comprising several search_outputs is returned.

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
                self.search_outputs[item]
            )
        return self.search_outputs[item]

    def __len__(self):
        return len(self.search_outputs)

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
            Objects representing predicates that may evaluate to `True` or `False` for any
            given phase output.

        Returns
        -------
        An aggregator comprising all search_outputs that evaluate to `True` for all predicates.
        """
        search_outputs = self.search_outputs
        for predicate in predicates:
            search_outputs = predicate.filter(search_outputs)
        search_outputs = list(search_outputs)
        print(f"Filter found a total of {str(len(search_outputs))} results")
        return AbstractAggregator(search_outputs=list(search_outputs))

    def values(self, name: str) -> Iterator:
        """
        Get values from outputs with a given name.

        A list the same length as the number of search_outputs is returned
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
            self.search_outputs
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
            self.search_outputs
        )

    def group_by(self, field: str) -> AggregatorGroup:
        """
        Group the search_outputs by a field, e.g. pipeline.

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
        for phase in self.search_outputs:
            group_dict[getattr(phase, field)].append(phase)
        return AggregatorGroup(list(map(AbstractAggregator, group_dict.values())))

    @property
    def model_results(self) -> str:
        """
        A string joining headers and results for all included search_outputs.
        """
        return "\n\n".join(
            "{}\n\n{}".format(phase.header, phase.model_results)
            for phase in self.search_outputs
        )


class Aggregator(AbstractAggregator):
    def __init__(
            self,
            directory: Union[str, os.PathLike],
            completed_only=False
    ):
        """
        Class to aggregate phase results for all subdirectories in a given directory.

        The whole directory structure is traversed and a Phase object created for each directory that contains a
        metadata file.

        Parameters
        ----------
        directory
            A directory in which the outputs of search_outputs are kept. This is searched recursively.
        completed_only
            If `True` only search_outputs with a .completed file (indicating the phase was completed)
            are included in the aggregator.
        """

        # TODO : Progress bar here

        print("Aggregator loading search_outputs... could take some time.")

        self._directory = directory
        search_outputs = []

        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(".zip"):
                    try:
                        with zipfile.ZipFile(path.join(root, filename), "r") as f:
                            f.extractall(path.join(root, filename[:-4]))
                    except zipfile.BadZipFile:
                        raise zipfile.BadZipFile(
                            f"File is not a zip file: \n "
                            f"{root} \n"
                            f"{filename}"
                        )

        for root, _, filenames in os.walk(directory):
            if "metadata" in filenames:
                if not completed_only or ".completed" in filenames:
                    search_outputs.append(SearchOutput(root))

        if len(search_outputs) == 0:
            print(f"\nNo search_outputs found in {directory}\n")
        else:
            print(f"\n A total of {str(len(search_outputs))} search_outputs and results were found.")
        super().__init__(search_outputs)
