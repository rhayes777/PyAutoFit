import json
import os
import shutil
from os import path
from typing import Optional

import dill

from autoconf import conf
from autofit.text import formatter
from autofit.tools.util import open_
from .abstract import AbstractPaths
from ..samples import load_from_table


class DirectoryPaths(AbstractPaths):
    def save_named_instance(
            self,
            name: str,
            instance
    ):
        """
        Save an instance, such as that at a given sigma
        """
        self.save_object(name, instance)

    def _path_for_pickle(
            self,
            name: str
    ):
        return path.join(
            self._pickle_path,
            f"{name}.pickle"
        )

    def save_object(
            self,
            name: str,
            obj: object
    ):
        """
        Serialise an object using dill and save it to the pickles
        directory of the search.

        Parameters
        ----------
        name
            The name of the object
        obj
            A serialisable object
        """
        with open_(
                self._path_for_pickle(
                    name
                ),
                "wb"
        ) as f:
            dill.dump(
                obj, f
            )

    def load_object(
            self,
            name: str
    ):
        """
        Load a serialised object with the given name.

        e.g. if the name is 'model' then pickles/model.pickle is loaded.

        Parameters
        ----------
        name
            The name of a serialised object

        Returns
        -------
        The deserialised object
        """
        with open_(
                self._path_for_pickle(
                    name
                ),
                "rb"
        ) as f:
            return dill.load(
                f
            )

    def remove_object(
            self,
            name: str
    ):
        """
        Remove the object with the given name from the pickles folder.

        Parameters
        ----------
        name
            The name of a pickle file excluding .pickle
        """
        try:
            os.remove(
                self._path_for_pickle(
                    name
                )
            )
        except FileNotFoundError:
            pass

    def is_object(
            self,
            name: str
    ) -> bool:
        """
        Is there a file pickles/{name}.pickle?
        """
        return os.path.exists(
            self._path_for_pickle(
                name
            )
        )

    @property
    def is_complete(self) -> bool:
        """
        Has the search been completed?
        """
        return path.exists(
            self._has_completed_path
        )

    def completed(self):
        """
        Mark the search as complete by saving a file
        """
        open_(self._has_completed_path, "w+").close()

    def load_samples(self):
        return load_from_table(
            filename=self._samples_file
        )

    def save_samples(self, samples):
        pass

    def samples_to_csv(self, samples):
        """
        Save the final-result samples associated with the phase as a pickle
        """
        if conf.instance["general"]["output"]["samples_to_csv"]:
            samples.write_table(filename=self._samples_file)
            samples.info_to_json(filename=self._info_file)

    def load_samples_info(self):
        with open_(self._info_file) as infile:
            return json.load(infile)

    def save_all(
            self,
            search_config_dict=None,
            info=None,
            pickle_files=None
    ):
        search_config_dict = search_config_dict or {}
        info = info or {}
        pickle_files = pickle_files or []

        self.save_identifier()
        self.save_parent_identifier()
        self._save_search(config_dict=search_config_dict)
        self._save_model_info(model=self.model)
        self._save_parameter_names_file(model=self.model)
        self.save_object("info", info)
        self.save_object("search", self.search)
        self.save_object("model", self.model)
        self._save_metadata(
            search_name=type(self.search).__name__.lower()
        )
        self._move_pickle_files(pickle_files=pickle_files)

    @AbstractPaths.parent.setter
    def parent(
            self,
            parent: AbstractPaths
    ):
        """
        The search performed before this search. For example, a search
        that is then compared to searches during a grid search.
        """
        self._parent = parent

    def save_parent_identifier(self):
        if self.parent is not None:
            with open_(self._parent_identifier_path, "w+") as f:
                f.write(
                    self.parent.identifier
                )
            self.parent.save_unique_tag()

    def save_unique_tag(
            self,
            is_grid_search=False
    ):
        if is_grid_search:
            with open_(self._grid_search_path, "w+") as f:
                if self.unique_tag is not None:
                    f.write(self.unique_tag)

    @property
    def _parent_identifier_path(self) -> str:
        return path.join(self.output_path, ".parent_identifier")

    @property
    def _grid_search_path(self) -> str:
        return path.join(self.output_path, ".is_grid_search")

    @property
    def is_grid_search(self) -> bool:
        """
        Is this a grid search which comprises a number of child searches?
        """
        return os.path.exists(
            self._grid_search_path
        )

    def create_child(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            is_identifier_in_paths: Optional[bool] = None,
            identifier: Optional[str] = None
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
            parent=self
        )
        child.model = self.model
        child.search = self.search
        child._identifier = identifier
        return child

    def for_sub_analysis(
            self,
            analysis_name: str
    ):
        """
        Paths for an analysis which is a child of another analysis.

        The analysis name forms a new directory on the end of the original
        analysis output path.
        """
        from .sub_directory_paths import SubDirectoryPaths
        return SubDirectoryPaths(
            parent=self,
            analysis_name=analysis_name
        )

    @property
    def _pickle_path(self) -> str:
        """
        This is private for a reason, use the save_object etc. methods to save and load pickles
        """
        return path.join(self.output_path, "pickles")

    def _save_metadata(self, search_name):
        """
        Save metadata associated with the phase, such as the name of the pipeline, the
        name of the phase and the name of the dataset being fit
        """
        with open_(path.join(self.output_path, "metadata"), "a") as f:
            f.write(f"""name={self.name}
            non_linear_search={search_name}
            """)

    def _move_pickle_files(self, pickle_files):
        """
        Move extra files a user has input the full path + filename of from the location specified to the
        pickles folder of the Aggregator, so that they can be accessed via the aggregator.
        """
        os.makedirs(
            self._pickle_path,
            exist_ok=True
        )
        if pickle_files is not None:
            [shutil.copy(file, self._pickle_path) for file in pickle_files]

    def _save_model_info(self, model):
        """
        Save the model.info file, which summarizes every parameter and prior.
        """
        with open_(path.join(
                self.output_path,
                "model.info"
        ), "w+") as f:
            f.write(f"Total Free Parameters = {model.prior_count} \n\n")
            f.write(f"{model.parameterization} \n\n")
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
        parameter_labels_with_subscript = [f"{label}_{subscript}" for label, subscript in
                                           zip(parameter_labels, subscripts)]

        parameter_name_and_label = []

        for i in range(model.prior_count):
            line = formatter.add_whitespace(
                str0=parameter_names[i], str1=parameter_labels_with_subscript[i], whitespace=70
            )
            parameter_name_and_label += [f"{line}\n"]

        formatter.output_list_of_strings_to_file(
            file=path.join(
                self.samples_path,
                "model.paramnames"
            ),
            list_of_strings=parameter_name_and_label
        )

    @property
    def _info_file(self) -> str:
        return path.join(self.samples_path, "info.json")

    @property
    def _has_completed_path(self) -> str:
        """
        A file indicating that a `NonLinearSearch` has been completed previously
        """
        return path.join(self.output_path, ".completed")

    def _make_path(self) -> str:
        """
        Returns the path to the folder at which the metadata should be saved

        The path terminates with the identifier, unless the identifier has already
        been added to the path.
        """
        path_ = path.join(
            conf.instance.output_path,
            self.path_prefix,
            self.name
        )
        if self.is_identifier_in_paths:
            path_ = path.join(
                path_,
                self.identifier
            )
        return path_
