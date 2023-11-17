import dill
import json
import logging
import os
import re
import shutil
import zipfile
from abc import ABC, abstractmethod
from configparser import NoSectionError
from os import path
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from autoconf import conf
from autofit.mapper.identifier import Identifier, IdentifierField
from autofit.text import text_util
from autofit.tools.util import open_, zip_directory

logger = logging.getLogger(__name__)

pattern = re.compile(r"(?<!^)(?=[A-Z])")


class AbstractPaths(ABC):
    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        is_identifier_in_paths=True,
        parent: Optional["AbstractPaths"] = None,
        unique_tag: Optional[str] = None,
        identifier: str = None,
    ):
        """
        Manages the path structure for `NonLinearSearch` output, for analyses both not using and using the search
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
        name
            The name of the non-linear search, which is used as a folder name after the ``path_prefix``. For searchs
            this name is the ``name``.
        path_prefix
            A prefixed path that appears after the output_path but before the name variable.
        is_identifier_in_paths
            If True output path and symlink path terminate with an identifier generated from the
            search and model
        """

        self.name = name or ""
        self.path_prefix = path_prefix or ""

        self.unique_tag = unique_tag

        self._non_linear_name = None
        self.__identifier = identifier or None

        self.is_identifier_in_paths = is_identifier_in_paths

        self._parent = None
        self.parent = parent

        try:
            self.remove_files = conf.instance["general"]["output"]["remove_files"]

            if conf.instance["general"]["hpc"]["hpc_mode"]:
                self.remove_files = True
        except NoSectionError as e:
            logger.exception(e)

    def save_parent_identifier(self):
        pass

    def save_unique_tag(self, is_grid_search=False):
        pass

    def __str__(self):
        return str(self.output_path)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    @property
    def parent(self) -> "AbstractPaths":
        """
        The search performed before this search. For example, a search
        that is then compared to searches during a grid search.
        """
        return self._parent

    @parent.setter
    @abstractmethod
    def parent(self, parent: "AbstractPaths"):
        pass

    @property
    @abstractmethod
    def is_grid_search(self) -> bool:
        pass

    def for_sub_analysis(self, analysis_name: str):
        return self.create_child(name=analysis_name)

    @abstractmethod
    def create_child(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        is_identifier_in_paths: Optional[bool] = None,
        identifier: Optional[str] = None,
    ) -> "AbstractPaths":
        """
        Create a paths object which is the child of some parent
        paths object. This is done during a GridSearch so that
        results can be stored in the correct directory.

        Parameters
        ----------
        name
        path_prefix
        is_identifier_in_paths
            If False then this path's identifier will not be
            added to its output path.
        identifier

        Returns
        -------
        A new paths object
        """

    search = IdentifierField()
    model = IdentifierField()
    unique_tag = IdentifierField()

    @property
    def non_linear_name(self):
        if self._non_linear_name is None:
            if self.search is not None:
                self._non_linear_name = pattern.sub(
                    "_", type(self.search).__name__
                ).lower()
        return self._non_linear_name

    @property
    def _identifier(self):
        if self.__identifier is None:
            if None in (self.model, self.search):
                logger.debug(
                    "Generating identifier without both model and search having been set."
                )

            identifier_list = [self.search, self.model]

            if self.unique_tag is not None:
                identifier_list.append(self.unique_tag)
            self.__identifier = Identifier(identifier_list)

        return self.__identifier

    @_identifier.setter
    def _identifier(self, identifier):
        self.__identifier = identifier

    @property
    def identifier(self):
        return str(self._identifier)

    def save_identifier(self):
        with open_(self.output_path / ".identifier", "w+") as f:
            f.write(self._identifier.description)

    @property
    def search_internal_path(self) -> Path:
        """
        The path to the samples folder.
        """

        os.makedirs(self._files_path / "search_internal", exist_ok=True)

        return self._files_path / "search_internal"

    @property
    def image_path(self) -> Path:
        """
        The path to the image folder.
        """
        return self.output_path / "image"

    @property
    def profile_path(self) -> Path:
        """
        The path to the profile folder.
        """
        return self.output_path / "profile"

    @property
    def output_path(self) -> Path:
        """
        The path to the output information for a search.
        """

        strings = list(
            filter(
                len,
                [
                    str(conf.instance.output_path),
                    str(self.path_prefix),
                    str(self.name),
                ],
            )
        )

        if self.is_identifier_in_paths:
            strings.append(self.identifier)

        return Path(path.join("", *strings))

    @property
    def _files_path(self) -> Path:
        """
        This is private for a reason, use the save_json etc. methods to save and load json
        """
        files_path = self.output_path / "files"
        os.makedirs(files_path, exist_ok=True)
        return files_path

    def zip_remove(self):
        """
        Copy files from the sym linked search folder then remove the sym linked folder.
        """

        self._zip()

    def _zip(self):
        try:
            zip_directory(self.output_path, self._zip_path)

            if self.remove_files:
                shutil.rmtree(
                    self.output_path,
                    ignore_errors=True,
                )

        except FileNotFoundError:
            pass

    def zip_remove_nuclear(self):
        """
        When multiple model-fits are performed using the same `path_prefix` and `name`,
        the results are populated in the same folder with different unique identifiers.

        By accident, one may perform runs where additional results are placed
        in these folders which are not wanted for the subsequent analysis.

        Removing these results from the directory can be cumbersome, as determining
        the unwanted results based on their unique identifier requires visually inspecting
        them.

        These unwanted results can also make manipulating the results via the database
        problematic, as one may need to again filter based on unique identifier.

        When a run is performed in nuclear mode, all results in every folder are
        deleted except the results corresponding to the unique identifier of that run.

        Therefore, provided the user is 100% certain that the run corresponds to the
        results they want to keep, nuclear mode can be used to remove all unwanted results.

        For example, suppose a folder has 5 results, 4 of which are unwanted and 1 which is
        wanted. If nuclear mode runs, and the model-fit is set up correctly such that the
        identifier created corresponds to the wanted result, all 4 unwanted results
        will be deleted.

        To enable nuclear mode, set the environment variable ``PYAUTOFIT_NUCLEAR_MODE=1``.

        Nuclear mode is dangerous, and must be used with CAUTION AND CARE!
        """

        if os.environ.get("PYAUTOFIT_NUCLEAR_MODE") == "1":
            file_path = Path(os.path.split(self.output_path)[0])

            file_list = os.listdir(file_path)
            file_list = [file for file in file_list if self.identifier not in file]

            for file in file_list:
                file_to_remove = file_path / file

                try:
                    os.remove(file_to_remove)
                    logger.info(f"NUCLEAR MODE -- Removed {file_to_remove}")
                except (IsADirectoryError, FileNotFoundError):
                    pass

                try:
                    shutil.rmtree(file_to_remove)
                    logger.info(f"NUCLEAR MODE -- Removed {file_to_remove}")
                except (NotADirectoryError, FileNotFoundError):
                    pass

    def restore(self):
        """
        Copy files from the ``.zip`` file to the samples folder.
        """

        if path.exists(self._zip_path):
            shutil.rmtree(self.output_path, ignore_errors=True)

            try:
                with zipfile.ZipFile(self._zip_path, "r") as f:
                    f.extractall(self.output_path)
            except zipfile.BadZipFile as e:
                raise zipfile.BadZipFile(
                    f"Unable to restore the zip file at the path {self._zip_path}"
                ) from e

            os.remove(self._zip_path)

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
    def save_json(self, name, object_dict: dict, prefix: str = ""):
        pass

    @abstractmethod
    def load_json(self, name, prefix: str = "") -> dict:
        pass

    @abstractmethod
    def save_array(self, name, array: np.ndarray):
        pass

    @abstractmethod
    def load_array(self, name) -> np.ndarray:
        pass

    @abstractmethod
    def save_fits(self, name: str, hdu, prefix: str = ""):
        pass

    @abstractmethod
    def load_fits(self, name: str, prefix: str = ""):
        pass

    @abstractmethod
    def save_object(self, name: str, obj: object, prefix: str = ""):
        pass

    @abstractmethod
    def load_object(self, name: str, prefix: str = ""):
        pass

    @abstractmethod
    def remove_object(self, name: str):
        pass

    @abstractmethod
    def is_object(self, name: str) -> bool:
        pass

    def save_search_internal(self, obj):
        raise NotImplementedError

    def load_search_internal(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def is_complete(self) -> bool:
        pass

    @abstractmethod
    def completed(self):
        pass

    @abstractmethod
    def save_all(self, search_config_dict=None, info=None):
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

    def samples_to_csv(self, samples):
        """
        Save the final-result samples associated with the phase as a pickle
        """

    @abstractmethod
    def load_samples_info(self):
        pass

    def save_summary(self, samples, log_likelihood_function_time):
        result_info = text_util.result_info_from(
            samples=samples,
        )

        filename = self.output_path / "model.results"

        with open_(filename, "w") as f:
            f.write(result_info)

        text_util.search_summary_to_file(
            samples=samples,
            log_likelihood_function_time=log_likelihood_function_time,
            filename=self.output_path / "search.summary",
        )

    @property
    def _samples_file(self) -> Path:
        return self._files_path / "samples.csv"

    @property
    def _covariance_file(self) -> Path:
        return self._files_path / "covariance.csv"

    @property
    def _info_file(self) -> Path:
        return self._files_path / "samples_info.json"
