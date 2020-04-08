import os
import pickle, dill

import autofit.optimize.non_linear.non_linear
from autofit.optimize.non_linear.output import AbstractOutput


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
        self.__optimizer = None
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
    def output(self) -> AbstractOutput:
        """
        An object describing the output data from the nonlinear search performed in this phase
        """
        return self.optimizer.output_from_model(
            model=self.model, paths=self.optimizer.paths
        )

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
                os.path.join(self.directory, f"dataset.pickle"), "rb"
        ) as f:
            return pickle.load(f)

    @property
    def mask(self):
        """
        A pickled mask object
        """
        with open(
                os.path.join(self.directory, "mask.pickle"), "rb"
        ) as f:
            return dill.load(f)

    @property
    def meta_dataset(self):
        """
        A pickled mask object
        """
        with open(
                os.path.join(self.directory, "meta_dataset.pickle"), "rb"
        ) as f:
            return pickle.load(f)

    @property
    def phase_attributes(self):
        """
        A pickled mask object
        """
        with open(
                os.path.join(self.directory, "phase_attributes.pickle"), "rb"
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
