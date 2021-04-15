import os
import re
import shutil
import zipfile
from abc import ABC, abstractmethod
from configparser import NoSectionError
from functools import wraps
from os import path
import json

from autoconf import conf
from autofit.mapper import link
from autofit.mapper.model_object import Identifier
from autofit.non_linear.log import logger
from autofit.text import text_util


def make_path(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        full_path = func(*args, **kwargs)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    return wrapper


pattern = re.compile(r'(?<!^)(?=[A-Z])')


class AbstractPaths(ABC):
    def __init__(
            self,
            name=None,
            path_prefix=None
    ):
        """Manages the path structure for `NonLinearSearch` output, for analyses both not using and using the search
        API. Use via non-linear searches requires manual input of paths, whereas the search API manages this using the
        search attributes.

        The output path within which the *Paths* objects path structure is contained is set via PyAutoConf, using the
        command:

        from autoconf import conf
        conf.instance = conf.Config(output_path="path/to/output")

        If we assume all the input strings above are used with the following example names:

        name = "name"
        path_prefix = "folder_0/folder_1"

        The output path of the `NonLinearSearch` results will be:

        /path/to/output/folder_0/folder_1/name

        Parameters
        ----------
        name : str
            The name of the non-linear search, which is used as a folder name after the ``path_prefix``. For searchs
            this name is the ``name``.
        path_prefix : str
            A prefixed path that appears after the output_path but beflore the name variable.
        """

        self.name = name or ""
        self.path_prefix = path_prefix or ""

        self._search = None
        self.model = None

        self._non_linear_name = None
        self._identifier = None

        try:
            self.remove_files = conf.instance["general"]["output"]["remove_files"]

            if conf.instance["general"]["hpc"]["hpc_mode"]:
                self.remove_files = True
        except NoSectionError as e:
            logger.exception(e)

    @property
    def search(self):
        return self._search

    @search.setter
    def search(self, search):
        self._search = search
        self._non_linear_name = pattern.sub(
            '_', type(
                self.search
            ).__name__
        ).lower()

    @property
    def non_linear_name(self):
        return self._non_linear_name

    @property
    def identifier(self):
        if None in (self.model, self.search):
            logger.warn(
                "Both model and search should be set"
            )
        if self._identifier is None:
            self._identifier = str(
                Identifier([
                    self.search,
                    self.model
                ])
            )
        return self._identifier

    @property
    def path(self):
        return link.make_linked_folder(self._sym_path)

    @property
    @make_path
    def samples_path(self) -> str:
        """
        The path to the samples folder.
        """
        return path.join(self.output_path, "samples")

    @property
    def image_path(self) -> str:
        """
        The path to the image folder.
        """
        return path.join(self.output_path, "image")

    @property
    @make_path
    def output_path(self) -> str:
        """
        The path to the output information for a search.
        """
        strings = (
            list(filter(
                len,
                [
                    str(conf.instance.output_path),
                    self.path_prefix,
                    self.name,
                    self.identifier,
                ],
            )
            )
        )

        return path.join("", *strings)

    def zip_remove(self):
        """
        Copy files from the sym linked search folder then remove the sym linked folder.
        """

        self._zip()

        if self.remove_files:
            try:
                shutil.rmtree(self.path)
            except (FileNotFoundError, PermissionError):
                pass

    def _zip(self):

        try:
            with zipfile.ZipFile(self._zip_path, "w", zipfile.ZIP_DEFLATED) as f:
                for root, dirs, files in os.walk(self.output_path):

                    for file in files:
                        f.write(
                            path.join(root, file),
                            path.join(
                                root[len(self.output_path):], file
                            ),
                        )

            if self.remove_files:
                shutil.rmtree(self.output_path)

        except FileNotFoundError:
            pass

    def restore(self):
        """
        Copy files from the ``.zip`` file to the samples folder.
        """

        if path.exists(self._zip_path):
            with zipfile.ZipFile(self._zip_path, "r") as f:
                f.extractall(self.output_path)

            os.remove(self._zip_path)

    @property
    @make_path
    def _sym_path(self) -> str:
        return path.join(
            conf.instance.output_path,
            self.path_prefix,
            self.name,
            self.identifier,
        )

    def __eq__(self, other):
        return isinstance(other, AbstractPaths) and all(
            [
                self.path_prefix == other.path_prefix,
                self.name == other.name,
                self.non_linear_name == other.non_linear_name,
            ]
        )

    @property
    def _zip_path(self) -> str:
        return f"{self.output_path}.zip"

    @abstractmethod
    def save_object(
            self,
            name: str,
            obj: object
    ):
        pass

    @abstractmethod
    def load_object(
            self,
            name: str
    ):
        pass

    @abstractmethod
    def remove_object(
            self,
            name: str
    ):
        pass

    @abstractmethod
    def is_object(
            self,
            name: str
    ) -> bool:
        pass

    @property
    @abstractmethod
    def is_complete(self) -> bool:
        pass

    @abstractmethod
    def completed(self):
        pass

    @abstractmethod
    def save_all(self, search_config_dict, info, pickle_files):
        pass

    @abstractmethod
    def load_samples(self):
        """
        Load samples from the database
        """

    @abstractmethod
    def save_samples(self, samples):
        """
        Save samples to the database
        """

    @abstractmethod
    def load_samples_info(self):
        pass

    def _save_search(self, config_dict):

        with open(path.join(self.output_path, "search.json"), "w+") as f:
            json.dump(config_dict, f, indent=4)

    def save_summary(self, samples, log_likelihood_function_time):
        text_util.results_to_file(
            samples=samples,
            filename=path.join(
                self.output_path,
                "model.results"
            )
        )

        text_util.search_summary_to_file(
            samples=samples,
            log_likelihood_function_time=log_likelihood_function_time,
            filename=path.join(
                self.output_path,
                "search.summary"
            )
        )

    @property
    def _samples_file(self) -> str:
        return path.join(self.samples_path, "samples.csv")

    @property
    def _info_file(self) -> str:
        return path.join(self.samples_path, "info.json")
