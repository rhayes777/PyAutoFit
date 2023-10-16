import csv
import json
import logging
import pickle
from os import path
from pathlib import Path
from typing import Generator, Tuple, Optional

import dill
import numpy as np

from autofit import SamplesPDF
from autofit.mapper.identifier import Identifier
from autofit.non_linear.samples.sample import samples_from_iterator
from autofit.non_linear.search import abstract_search
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autoconf.dictable import from_dict

original_create_file_handle = dill._dill._create_filehandle


def _create_file_handle(*args, **kwargs):
    """
    Handle FileNotFoundError when attempting to deserialize pickles
    using dill and return None instead.
    """
    try:
        return original_create_file_handle(*args, **kwargs)
    except pickle.UnpicklingError as e:
        if not isinstance(e.args[0], FileNotFoundError):
            raise e
        logging.warning(
            f"Could not create a handler for {e.args[0].filename} as it does not exist"
        )
        return None


dill._dill._create_filehandle = _create_file_handle


class SearchOutput:
    """
    @DynamicAttrs
    """

    def __new__(cls, directory: Path, reference: dict = None):
        """
        Represents the output of a single search. Comprises a metadata file and other dataset files.

        Parameters
        ----------
        directory
            The directory of the search
        """
        if (directory / ".is_grid_search").exists():
            return super().__new__(GridSearchOutput)
        return super().__new__(cls)

    def __init__(self, directory: Path, reference: dict = None):
        """
        Represents the output of a single search. Comprises a metadata file and other dataset files.

        Parameters
        ----------
        directory
            The directory of the search
        """
        self.__search = None
        self.__model = None

        self.directory = directory

        self._reference = reference
        self.file_path = directory / "metadata"

        try:
            with open(self.file_path) as f:
                self.text = f.read()
                pairs = [
                    line.split("=") for line in self.text.split("\n") if "=" in line
                ]
                self.__dict__.update({pair[0]: pair[1] for pair in pairs})
        except FileNotFoundError:
            pass

    @property
    def files_path(self):
        return self.directory / "files"

    def __getattr__(self, item):
        """
        Attempt to load a pickle by the same name from the search output directory.

        dataset.pickle, meta_dataset.pickle etc.
        """
        try:
            with open(self.files_path / f"{item}.pickle", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            pass
        try:
            with open(self.files_path / f"{item}.json") as f:
                d = json.load(f)
                if "type" in d:
                    result = from_dict(d, reference=self._reference)
                    if result is not None:
                        return result
                return d
        except FileNotFoundError:
            pass
        try:
            with open(self.files_path / f"{item}.csv") as f:
                return np.loadtxt(f)
        except (FileNotFoundError, ValueError):
            pass

    @property
    def id(self):
        return str(Identifier([self.search, self.model, self.search.unique_tag]))

    @property
    def is_complete(self) -> bool:
        """
        Whether the search has completed
        """
        return (self.directory / ".completed").exists()

    @property
    def samples(self):
        try:
            info_path = self.files_path / "samples_info.json"
            samples_path = self.files_path / "samples.csv"
            with open(info_path) as f:
                info_json = json.load(f)
            with open(samples_path) as f:
                sample_list = samples_from_iterator(csv.reader(f))

            return SamplesPDF.from_list_info_and_model(
                sample_list=sample_list,
                samples_info=info_json,
                model=self.model,
            )
        except FileNotFoundError:
            raise AttributeError("No samples found")

    def names_and_paths(
        self,
        suffix: str,
    ) -> Generator[Tuple[str, Path], None, None]:
        """
        Get the names and paths of files with a given suffix.

        Parameters
        ----------
        suffix
            The suffix of the files to retrieve (e.g. ".json")

        Returns
        -------
        A generator of tuples of the form (name, path) where name is the path to the file
        joined by . without the suffix and path is the path to the file
        """
        for file in list(self.files_path.rglob(f"*{suffix}")):
            name = ".".join(file.relative_to(self.files_path).with_suffix("").parts)
            yield name, file

    @property
    def child_analyses(self):
        """
        A list of child analyses loaded from the analyses directory
        """
        return list(map(SearchOutput, Path(self.directory).glob("analyses/*")))

    @property
    def model_results(self) -> str:
        """
        Reads the model.results file
        """
        with open(self.directory / "model.results") as f:
            return f.read()

    @property
    def mask(self):
        """
        A pickled mask object
        """
        with open(self.files_path / "mask.pickle", "rb") as f:
            return dill.load(f)

    @property
    def header(self) -> str:
        """
        A header created by joining the search name
        """
        phase = self.phase or ""
        dataset_name = self.dataset_name or ""
        return path.join(phase, dataset_name)

    @property
    def search(self) -> abstract_search.NonLinearSearch:
        """
        The search object that was used in this phase
        """
        if self.__search is None:
            try:
                with open(self.files_path / "search.json") as f:
                    self.__search = from_dict(json.load(f))
            except (FileNotFoundError, ModuleNotFoundError):
                try:
                    with open(self.files_path / "search.pickle", "rb") as f:
                        self.__search = pickle.load(f)
                except (FileNotFoundError, ModuleNotFoundError):
                    logging.warning("Could not load search")
        return self.__search

    def child_values(self, name):
        """
        Get the values of a given key for all children
        """
        return [getattr(child, name) for child in self.child_analyses]

    def __str__(self):
        return self.text

    def __repr__(self):
        return "<PhaseOutput {}>".format(self)


class GridSearchOutput(SearchOutput):
    @property
    def parent_identifier(self) -> Optional[str]:
        """
        Read the parent identifier for a fit in a directory.

        Defaults to None if no .parent_identifier file is found.
        """
        try:
            return (self.directory / ".parent_identifier").read_text()
        except FileNotFoundError:
            return None
