import os
import pickle

import dill

from autofit.non_linear import abstract_search
from autofit.non_linear.samples import PDFSamples


class PhaseOutput:
    """
    @DynamicAttrs
    """

    def __init__(self, directory: str):
        """
        Represents the output of a single phase. Comprises a metadata file and other dataset files.

        Parameters
        ----------
        directory
            The directory of the phase
        """
        self.directory = directory
        self.__search = None
        self.__model = None
        self.file_path = os.path.join(directory, "metadata")
        with open(self.file_path) as f:
            self.text = f.read()
            pairs = [
                line.split("=")
                for line
                in self.text.split("\n")
                if "=" in line
            ]
            self.__dict__.update({pair[0]: pair[1] for pair in pairs})

    @property
    def pickle_path(self):
        return f"{self.directory}/pickles"

    @property
    def model_results(self) -> str:
        """
        Reads the model.results file
        """
        with open(os.path.join(self.directory, "model.results")) as f:
            return f.read()

    @property
    def mask(self):
        """
        A pickled mask object
        """
        with open(
                os.path.join(self.pickle_path, "mask.pickle"), "rb"
        ) as f:
            return dill.load(f)

    def __getattr__(self, item):
        """
        Attempt to load a pickle by the same name from the phase output directory.

        dataset.pickle, meta_dataset.pickle etc.
        """
        try:
            with open(
                    os.path.join(self.pickle_path, f"{item}.pickle"), "rb"
            ) as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"No {item} associated with {self.directory}")

    @property
    def header(self) -> str:
        """
        A header created by joining the pipeline, phase and dataset names
        """
        return "/".join((self.pipeline, self.phase, self.dataset_name))

    @property
    def search(self) -> abstract_search.NonLinearSearch:
        """
        The search object that was used in this phase
        """
        if self.__search is None:
            with open(os.path.join(self.pickle_path, "search.pickle"), "r+b") as f:
                self.__search = pickle.loads(f.read())
        return self.__search

    @property
    def model(self):
        """
        The model that was used in this phase
        """
        if self.__model is None:
            with open(os.path.join(self.pickle_path, "model.pickle"), "r+b") as f:
                self.__model = pickle.loads(f.read())
        return self.__model

    def __str__(self):
        return self.text

    def __repr__(self):
        return "<PhaseOutput {}>".format(self)
