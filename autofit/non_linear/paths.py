import json
import os
import re
import shutil
import zipfile
from configparser import NoSectionError
from functools import wraps
from os import path

import dill

from autoconf import conf
from autofit.mapper import link
from autofit.mapper.model_object import Identifier
from autofit.non_linear import samples as s
from autofit.non_linear.log import logger
from autofit.text import formatter, text_util


def make_path(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        full_path = func(*args, **kwargs)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    return wrapper


pattern = re.compile(r'(?<!^)(?=[A-Z])')


class Paths:
    def __init__(
            self,
            name="",
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
        tag = "tag"
        path_prefix = "folder_0/folder_1"
        non_linear_name = "emcee"

        The output path of the `NonLinearSearch` results will be:

        /path/to/output/folder_0/folder_1/name/tag/emcee

        Parameters
        ----------
        name : str
            The name of the non-linear search, which is used as a folder name after the ``path_prefix``. For searchs
            this name is the ``name``.
        path_prefix : str
            A prefixed path that appears after the output_path but beflore the name variable.
        """

        self.path_prefix = path_prefix or ""
        self.name = name or ""

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

    def save_samples(self, samples):
        """
        Save the final-result samples associated with the phase as a pickle
        """
        samples.write_table(filename=self._samples_file)
        samples.info_to_json(filename=self._info_file)

        self.save_object(
            "samples",
            samples
        )

    @property
    def identifier(self):
        if None in (self.model, self.search):
            logger.warn(
                "Both model and search should be set before the tag is determined"
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
        with open(
                self._path_for_pickle(
                    name
                ),
                "w+b"
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
        with open(
                self._path_for_pickle(
                    name
                ),
                "r+b"
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
        open(self._has_completed_path, "w+").close()

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

    def restore(self):
        """
        Copy files from the ``.zip`` file to the samples folder.
        """

        if path.exists(self._zip_path):
            with zipfile.ZipFile(self._zip_path, "r") as f:
                f.extractall(self.output_path)

            os.remove(self._zip_path)

    def load_samples(self):
        return s.load_from_table(
            filename=self._samples_file
        )

    def load_samples_info(self):
        with open(self._info_file) as infile:
            return json.load(infile)

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

    def save_all(self, info, pickle_files):
        self._save_model_info(model=self.model)
        self._save_parameter_names_file(model=self.model)
        self.save_object("info", info)
        self.save_object("search", self.search)
        self.save_object("model", self.model)
        self._save_metadata(
            search_name=type(self.search).__name__.lower()
        )
        self._move_pickle_files(pickle_files=pickle_files)

    @property
    @make_path
    def _pickle_path(self) -> str:
        """
        This is private for a reason - use the save_object etc. methods to save and load pickles
        """
        return path.join(self._make_path(), "pickles")

    @property
    def _zip_path(self) -> str:
        return f"{self.output_path}.zip"

    def _save_metadata(self, search_name):
        """
        Save metadata associated with the phase, such as the name of the pipeline, the
        name of the phase and the name of the dataset being fit
        """
        with open(path.join(self._make_path(), "metadata"), "a") as f:
            f.write(f"""name={self.name}
non_linear_search={search_name}
""")

    def _move_pickle_files(self, pickle_files):
        """
        Move extra files a user has input the full path + filename of from the location specified to the
        pickles folder of the Aggregator, so that they can be accessed via the aggregator.
        """
        if pickle_files is not None:
            [shutil.copy(file, self._pickle_path) for file in pickle_files]

    def _save_model_info(self, model):
        """Save the model.info file, which summarizes every parameter and prior."""
        with open(path.join(
                self.output_path,
                "model.info"
        ), "w+") as f:
            f.write(f"Total Free Parameters = {model.prior_count} \n\n")
            f.write(model.info)

    def _save_parameter_names_file(self, model):
        """Create the param_names file listing every parameter's label and Latex tag, which is used for *corner.py*
        visualization.

        The parameter labels are determined using the label.ini and label_format.ini config files."""

        parameter_names = model.model_component_and_parameter_names
        parameter_labels = model.parameter_labels
        subscripts = model.subscripts
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
        return isinstance(other, Paths) and all(
            [
                self.path_prefix == other.path_prefix,
                self.name == other.name,
                self.non_linear_name == other.non_linear_name,
            ]
        )

    @property
    def _samples_file(self) -> str:
        return path.join(self.samples_path, "samples.csv")

    @property
    def _info_file(self) -> str:
        return path.join(self.samples_path, "info.json")

    @property
    def _has_completed_path(self) -> str:
        """
        A file indicating that a `NonLinearSearch` has been completed previously
        """
        return path.join(self.output_path, ".completed")

    @make_path
    def _make_path(self) -> str:
        """
        Returns the path to the folder at which the metadata should be saved
        """
        return path.join(
            conf.instance.output_path,
            self.path_prefix,
            self.name,
            self.identifier,
        )
