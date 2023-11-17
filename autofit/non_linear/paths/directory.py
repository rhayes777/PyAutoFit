import dill
import json
import os
from os import path
from pathlib import Path
from typing import Optional, Union
import logging

from autoconf import conf
from autoconf.dictable import to_dict
from autoconf.output import conditional_output
from autofit.text import formatter
from autofit.tools.util import open_

from .abstract import AbstractPaths
from ..samples import load_from_table
from autofit.non_linear.samples.pdf import SamplesPDF
import numpy as np

logger = logging.getLogger(__name__)


class DirectoryPaths(AbstractPaths):
    def _path_for_pickle(self, name: str, prefix: str = "") -> Path:
        return self._files_path / prefix / f"{name}.pickle"

    def _path_for_json(self, name, prefix: str = "") -> Path:
        return self._files_path / prefix / f"{name}.json"

    def _path_for_csv(self, name) -> Path:
        return self._files_path / f"{name}.csv"

    def _path_for_fits(self, name, prefix: str = "") -> Path:
        os.makedirs(self._files_path / prefix, exist_ok=True)

        return self._files_path / prefix / f"{name}.fits"

    @conditional_output
    def save_object(self, name: str, obj: object, prefix: str = ""):
        """
        Serialise an object using dill and save it to the pickles
        directory of the search.

        Parameters
        ----------
        name
            The name of the object
        obj
            A serialisable object
        prefix
            A prefix to add to the path which is the name of the folder the file is saved in.
        """
        with open_(self._path_for_pickle(name, prefix), "wb") as f:
            dill.dump(obj, f)

    @conditional_output
    def save_json(self, name, object_dict: Union[dict, list], prefix: str = ""):
        """
        Save a dictionary as a json file in the jsons directory of the search.

        Parameters
        ----------
        name
            The name of the json file
        object_dict
            The dictionary to save
        prefix
            A prefix to add to the path which is the name of the folder the file is saved in.
        """
        with open_(self._path_for_json(name, prefix), "w+") as f:
            json.dump(object_dict, f, indent=4)

    def load_json(self, name, prefix: str = ""):
        with open_(self._path_for_json(name, prefix)) as f:
            return json.load(f)

    @conditional_output
    def save_array(self, name: str, array: np.ndarray):
        """
        Save a numpy array as a csv file in the csvs directory of the search.

        Parameters
        ----------
        name
            The name of the csv file
        array
            The numpy array to save
        """
        # noinspection PyTypeChecker
        np.savetxt(self._path_for_csv(name), array, delimiter=",")

    def load_array(self, name: str):
        return np.loadtxt(self._path_for_csv(name), delimiter=",")

    @conditional_output
    def save_fits(self, name: str, hdu, prefix: str = ""):
        """
        Save an HDU as a fits file in the fits directory of the search.

        Parameters
        ----------
        name
            The name of the fits file
        hdu
            The HDU to save
        prefix
            A prefix to add to the path which is the name of the folder the file is saved in.
        """
        hdu.writeto(self._path_for_fits(name, prefix), overwrite=True)

    def load_fits(self, name: str, prefix: str = ""):
        """
        Load an HDU from a fits file in the fits directory of the search.

        Parameters
        ----------
        name
            The name of the fits file
        prefix
            A prefix to add to the path which is the name of the folder the file is saved in.

        Returns
        -------
        The loaded HDU.
        """
        from astropy.io import fits

        return fits.open(self._path_for_fits(name, prefix))[0]

    def load_object(self, name: str, prefix: str = ""):
        """
        Load a serialised object with the given name.

        e.g. if the name is 'model' then pickles/model.pickle is loaded.

        Parameters
        ----------
        name
            The name of a serialised object
        prefix
            A prefix to add to the path which is the name of the folder the file is saved in.

        Returns
        -------
        The deserialised object
        """
        with open_(self._path_for_pickle(name, prefix), "rb") as f:
            return dill.load(f)

    def remove_object(self, name: str):
        """
        Remove the object with the given name from the pickles folder.

        Parameters
        ----------
        name
            The name of a pickle file excluding .pickle
        """
        try:
            os.remove(self._path_for_pickle(name))
        except FileNotFoundError:
            pass

    def is_object(self, name: str) -> bool:
        """
        Is there a file pickles/{name}.pickle?
        """
        return os.path.exists(self._path_for_pickle(name))

    @property
    def is_complete(self) -> bool:
        """
        Has the search been completed?
        """
        return path.exists(self._has_completed_path)

    def save_search_internal(self, obj):
        """
        Save the internal representation of a non-linear search as dill file.

        The results in this representation are required to use a search's in-built tools for visualization,
        analysing samples and other tasks.
        """
        filename = self.search_internal_path / "search_internal.dill"

        with open_(filename, "wb") as f:
            dill.dump(obj, f)

    def load_search_internal(self):
        """
        Load the internal representation of a non-linear search from a pickle or dill file.

        The results in this representation are required to use a search's in-built tools for visualization,
        analysing samples and other tasks.

        Returns
        -------
        The results of the non-linear search in its internal representation.
        """
        filename = self.search_internal_path / "search_internal.dill"

        with open_(filename, "rb") as f:
            return dill.load(f)

    def completed(self):
        """
        Mark the search as complete by saving a file
        """
        open_(self._has_completed_path, "w+").close()

    def load_samples(self):
        return load_from_table(filename=self._samples_file)

    def save_samples(self, samples):
        pass

    def samples_to_csv(self, samples):
        """
        Save the final-result samples associated with the phase as a pickle
        """
        if conf.instance["general"]["output"]["samples_to_csv"]:
            samples.write_table(filename=self._samples_file)
            self.save_json("samples_info", samples.samples_info)
            if isinstance(samples, SamplesPDF):
                try:
                    samples.save_covariance_matrix(self._covariance_file)
                except ValueError as e:
                    logger.warning(
                        f"Could not save covariance matrix because of the following error:\n{e}"
                    )

    def load_samples_info(self):
        with open_(self._info_file) as infile:
            return json.load(infile)

    def save_all(self, search_config_dict=None, info=None):
        info = info or {}

        self.save_identifier()
        self.save_parent_identifier()
        self._save_model_info(model=self.model)
        self._save_parameter_names_file(model=self.model)
        if info:
            self.save_json("info", info)
        self.save_json("search", to_dict(self.search))
        self.save_json("model", to_dict(self.model))
        self._save_metadata(search_name=type(self.search).__name__.lower())

    @AbstractPaths.parent.setter
    def parent(self, parent: AbstractPaths):
        """
        The search performed before this search. For example, a search
        that is then compared to searches during a grid search.
        """
        self._parent = parent

    def save_parent_identifier(self):
        if self.parent is not None:
            with open_(self._parent_identifier_path, "w+") as f:
                f.write(self.parent.identifier)
            self.parent.save_unique_tag()

    def save_unique_tag(self, is_grid_search=False):
        if is_grid_search:
            with open_(self._grid_search_path, "w+") as f:
                if self.unique_tag is not None:
                    f.write(self.unique_tag)

    @property
    def _parent_identifier_path(self) -> Path:
        return self.output_path / ".parent_identifier"

    @property
    def _grid_search_path(self) -> Path:
        return self.output_path / ".is_grid_search"

    @property
    def is_grid_search(self) -> bool:
        """
        Is this a grid search which comprises a number of child searches?
        """
        return os.path.exists(self._grid_search_path)

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
        child = type(self)(
            name=name or self.name,
            path_prefix=path_prefix or self.path_prefix,
            is_identifier_in_paths=(
                is_identifier_in_paths
                if is_identifier_in_paths is not None
                else self.is_identifier_in_paths
            ),
            parent=self,
        )
        child.model = self.model
        child.search = self.search
        child._identifier = identifier
        return child

    def for_sub_analysis(self, analysis_name: str):
        """
        Paths for an analysis which is a child of another analysis.

        The analysis name forms a new directory on the end of the original
        analysis output path.
        """
        from .sub_directory_paths import SubDirectoryPaths

        return SubDirectoryPaths(parent=self, analysis_name=analysis_name)

    def _save_metadata(self, search_name):
        """
        Save metadata associated with the phase, such as the name of the pipeline, the
        name of the phase and the name of the dataset being fit
        """
        with open_(self.output_path / "metadata", "a") as f:
            f.write(
                f"""name={self.name}\nnon_linear_search={search_name}
            """
            )

    def _save_model_info(self, model):
        """
        Save the model.info file, which summarizes every parameter and prior.
        """
        with open_(self.output_path / "model.info", "w+") as f:
            f.write(model.info)

    def _save_parameter_names_file(self, model):
        """
        Create the param_names file listing every parameter's label and Latex tag, which is used for corner.py
        visualization.

        The parameter labels are determined using the label.ini and label_format.ini config files.
        """

        parameter_names = model.model_component_and_parameter_names
        parameter_labels = model.parameter_labels
        subscripts = model.superscripts_overwrite_via_config
        parameter_labels_with_subscript = [
            f"{label}_{subscript}"
            for label, subscript in zip(parameter_labels, subscripts)
        ]

        parameter_name_and_label = []

        for i in range(model.prior_count):
            line = formatter.add_whitespace(
                str0=parameter_names[i],
                str1=parameter_labels_with_subscript[i],
                whitespace=70,
            )
            parameter_name_and_label += [f"{line}\n"]

        formatter.output_list_of_strings_to_file(
            file=self._files_path / "model.paramnames",
            list_of_strings=parameter_name_and_label,
        )

    @property
    def _info_file(self) -> Path:
        return self._files_path / "samples_info.json"

    @property
    def _has_completed_path(self) -> Path:
        """
        A file indicating that a `NonLinearSearch` has been completed previously
        """
        return self.output_path / ".completed"

    def _make_path(self) -> str:
        """
        Returns the path to the folder at which the metadata should be saved

        The path terminates with the identifier, unless the identifier has already
        been added to the path.
        """
        path_ = Path(path.join(conf.instance.output_path, self.path_prefix, self.name))
        if self.is_identifier_in_paths:
            path_ = path_ / self.identifier
        return path_
