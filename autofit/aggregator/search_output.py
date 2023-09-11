import json
import logging
import pickle
from os import path
from pathlib import Path

import dill
import numpy as np

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


class Output:
    def __init__(self, directory: Path):
        self.directory = directory

    @property
    def pickle_path(self):
        return self.directory / "pickles"

    @property
    def files_path(self):
        return self.directory / "files"

    def __getattr__(self, item):
        """
        Attempt to load a pickle by the same name from the search output directory.

        dataset.pickle, meta_dataset.pickle etc.
        """
        try:
            with open(self.pickle_path / f"{item}.pickle", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            pass
        try:
            with open(self.files_path / f"{item}.json") as f:
                return json.load(f)
        except FileNotFoundError:
            pass
        try:
            with open(self.files_path / f"{item}.csv") as f:
                return np.loadtxt(f)
        except (FileNotFoundError, ValueError):
            pass


class SearchOutput(Output):
    """
    @DynamicAttrs
    """

    def __init__(self, directory: str, reference: dict = None):
        """
        Represents the output of a single search. Comprises a metadata file and other dataset files.

        Parameters
        ----------
        directory
            The directory of the search
        """
        directory = Path(directory)
        super().__init__(directory)
        self.__search = None
        self.__model = None

        self._reference = reference
        self.file_path = directory / "metadata"

        with open(self.file_path) as f:
            self.text = f.read()
            pairs = [line.split("=") for line in self.text.split("\n") if "=" in line]
            self.__dict__.update({pair[0]: pair[1] for pair in pairs})

    @property
    def child_analyses(self):
        """
        A list of child analyses loaded from the analyses directory
        """
        return list(map(Output, Path(self.directory).glob("analyses/*")))

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
        with open(self.pickle_path / "mask.pickle", "rb") as f:
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
                    with open(self.pickle_path / "search.pickle", "rb") as f:
                        self.__search = pickle.load(f)
                except (FileNotFoundError, ModuleNotFoundError):
                    logging.warning("Could not load search")
        return self.__search

    def child_values(self, name):
        """
        Get the values of a given key for all children
        """
        return [getattr(child, name) for child in self.child_analyses]

    @property
    def model(self):
        """
        The model that was used in this phase
        """
        if self.__model is None:
            try:
                with open(self.files_path / "model.json") as f:
                    self.__model = AbstractPriorModel.from_dict(
                        json.load(f),
                        reference=self._reference,
                    )
            except (FileNotFoundError, ModuleNotFoundError) as e:
                logging.exception(e)
                try:
                    with open(self.pickle_path / "model.pickle", "rb") as f:
                        self.__model = pickle.load(f)
                except (FileNotFoundError, ModuleNotFoundError):
                    logging.warning("Could not load model")
        return self.__model

    def __str__(self):
        return self.text

    def __repr__(self):
        return "<PhaseOutput {}>".format(self)
