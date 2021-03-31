import json
import os
import pickle
import shutil
import zipfile
from configparser import NoSectionError
from functools import wraps
from os import path

from autoconf import conf
from autofit.mapper import link
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


class Paths:
    def __init__(
            self,
            name="",
            tag=None,
            path_prefix=None,
            non_linear_name=None,
            non_linear_tag_function=lambda: ""
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
        tag : str
            A tag for the non-linear search, typically used for instances where the same data is fitted with the same
            model but with slight variants. For searchs this is the search_tag.
        path_prefix : str
            A prefixed path that appears after the output_path but beflore the name variable.
        non_linear_name : str
            The name of the non-linear search, e.g. Emcee -> emcee. searchs automatically set up and use this variable.
        """

        self.path_prefix = path_prefix or ""
        self.name = name or ""
        self.tag = tag or ""
        self.non_linear_name = non_linear_name or ""
        self.non_linear_tag_function = non_linear_tag_function

        try:
            self.remove_files = conf.instance["general"]["output"]["remove_files"]

            if conf.instance["general"]["hpc"]["hpc_mode"]:
                self.remove_files = True
        except NoSectionError as e:
            logger.exception(e)

    def save_samples(self, samples):
        """
        Save the final-result samples associated with the phase as a pickle
        """
        samples.write_table(filename=self._samples_file)
        samples.info_to_json(filename=self._info_file)

        with open(path.join(
                self.pickle_path,
                "samples.pickle"
        ), "w+b") as f:
            f.write(pickle.dumps(samples))

    @property
    def non_linear_tag(self):
        return self.non_linear_tag_function()

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
    def zip_path(self) -> str:
        return f"{self.output_path}.zip"

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
                    self.tag,
                    self.non_linear_tag,
                ],
            )
            )
        )

        return path.join("", *strings)

    @property
    def is_complete(self):
        return path.exists(
            self._has_completed_path
        )

    def completed(self):
        open(self._has_completed_path, "w+").close()

    @property
    @make_path
    def pickle_path(self) -> str:
        return path.join(self._make_path(), "pickles")

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

        if path.exists(self.zip_path):
            with zipfile.ZipFile(self.zip_path, "r") as f:
                f.extractall(self.output_path)

            os.remove(self.zip_path)

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

    def save_all(self, model, info, search, pickle_files):
        self._save_model_info(model=model)
        self._save_parameter_names_file(model=model)
        self._save_info(info=info)
        self._save_search(search=search)
        self._save_model(model=model)
        self._save_metadata(
            search_name=type(self).__name__.lower()
        )
        self._move_pickle_files(pickle_files=pickle_files)

    def _save_metadata(self, search_name):
        """
        Save metadata associated with the phase, such as the name of the pipeline, the
        name of the phase and the name of the dataset being fit
        """
        with open(path.join(self._make_path(), "metadata"), "a") as f:
            f.write(f"""name={self.name}
tag={self.tag}
non_linear_search={search_name}
""")

    def _move_pickle_files(self, pickle_files):
        """
        Move extra files a user has input the full path + filename of from the location specified to the
        pickles folder of the Aggregator, so that they can be accessed via the aggregator.
        """
        if pickle_files is not None:
            [shutil.copy(file, self.pickle_path) for file in pickle_files]

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

    def _save_info(self, info):
        """
        Save the dataset associated with the phase
        """
        with open(path.join(self.pickle_path, "info.pickle"), "wb") as f:
            pickle.dump(info, f)

    def _save_search(self, search):
        """
        Save the search associated with the phase as a pickle
        """
        with open(path.join(
                self.pickle_path,
                "search.pickle"
        ), "w+b") as f:
            f.write(pickle.dumps(search))

    def _zip(self):

        try:
            with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as f:
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

    def _save_model(self, model):
        """
        Save the model associated with the phase as a pickle
        """
        with open(path.join(
                self.pickle_path,
                "model.pickle"
        ), "w+b") as f:
            f.write(pickle.dumps(model))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["non_linear_tag"] = state.pop("non_linear_tag_function")()
        return state

    def __setstate__(self, state):
        non_linear_tag = state.pop("non_linear_tag")
        self.non_linear_tag_function = lambda: non_linear_tag
        self.__dict__.update(state)

    @property
    @make_path
    def _sym_path(self) -> str:
        return path.join(
            conf.instance.output_path,
            self.path_prefix,
            self.name,
            self.tag,
            self.non_linear_tag,
        )

    def __eq__(self, other):
        return isinstance(other, Paths) and all(
            [
                self.path_prefix == other.path_prefix,
                self.name == other.name,
                self.tag == other.tag,
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
            self.tag,
            self.non_linear_tag,
        )
