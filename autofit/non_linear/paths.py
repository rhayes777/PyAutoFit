import os
from os import path
import shutil
import zipfile
from configparser import NoSectionError
from functools import wraps
import copy

from autoconf import conf
from autofit.mapper import link
from autofit.non_linear.log import logger


def make_path(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        full_path = func(*args, **kwargs)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    return wrapper


def convert_paths(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(args) > 1:
            raise AssertionError(
                "Only phase name is allowed to be a positional argument in a phase constructor"
            )

        first_arg = kwargs.pop("paths", None)
        if first_arg is None and len(args) == 1:
            first_arg = args[0]

        if isinstance(first_arg, Paths):
            return func(self, paths=first_arg, **kwargs)

        if first_arg is None:
            first_arg = kwargs.pop("name", None)

        # TODO : Using the class nam avoids us needing to mak an sintance - still cant get the kwargs.get() to work
        # TODO : nicely though.

        search = kwargs.get("search")

        if search is not None:

            search = kwargs["search"]
            search_name = search._config("tag", "name", str)

            def non_linear_tag_function():
                return search.tag

        else:

            search_name = None

            def non_linear_tag_function():
                return ""

        paths = Paths(
            name=first_arg,
            tag=kwargs.pop("phase_tag", None),
            path_prefix=kwargs.pop("path_prefix", None),
            non_linear_name=search_name,
            non_linear_tag_function=non_linear_tag_function,
        )

        if search is not None:
            search.paths = paths

        func(self, paths=paths, **kwargs)

    return wrapper


class Paths:
    def __init__(
            self,
            name="",
            tag=None,
            path_prefix=None,
            non_linear_name=None,
            non_linear_tag_function=lambda: "",
            remove_files=False,
    ):
        """Manages the path structure for `NonLinearSearch` output, for analyses both not using and using the phase
        API. Use via non-linear searches requires manual input of paths, whereas the phase API manages this using the
        phase attributes.

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
            The name of the non-linear search, which is used as a folder name after the ``path_prefix``. For phases
            this name is the ``name``.
        tag : str
            A tag for the non-linear search, typically used for instances where the same data is fitted with the same
            model but with slight variants. For phases this is the phase_tag.
        path_prefix : str
            A prefixed path that appears after the output_path but beflore the name variable.
        non_linear_name : str
            The name of the non-linear search, e.g. Emcee -> emcee. Phases automatically set up and use this variable.
        remove_files : bool
            If `True`, all output results except their ``.zip`` files are removed. If `False` they are not removed.
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

    def __getstate__(self):
        state = self.__dict__.copy()
        state["non_linear_tag"] = state.pop("non_linear_tag_function")()
        return state

    def __setstate__(self, state):
        non_linear_tag = state.pop("non_linear_tag")
        self.non_linear_tag_function = lambda: non_linear_tag
        self.__dict__.update(state)

    @property
    def non_linear_tag(self):
        return self.non_linear_tag_function()

    @property
    def path(self):
        return link.make_linked_folder(self.sym_path)

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
    def samples_path(self) -> str:
        """
        The path to the samples folder.
        """
        return path.join(self.output_path, "samples")

    @property
    def samples_file(self) -> str:
        return path.join(self.samples_path, "samples.csv")

    @property
    def info_file(self) -> str:
        return path.join(self.samples_path, "info.json")

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
        The path to the output information for a phase.
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
    def has_completed_path(self) -> str:
        """
        A file indicating that a `NonLinearSearch` has been completed previously
        """
        return path.join(self.output_path, ".completed")

    @property
    def execution_time_path(self) -> str:
        """
        The path to the output information for a phase.
        """
        return path.join(self.name_folder, "execution_time")

    @property
    @make_path
    def name_folder(self):
        return path.join(conf.instance.output_path, self.path_prefix, self.name)

    @property
    @make_path
    def sym_path(self) -> str:
        return path.join(
            conf.instance.output_path,
            self.path_prefix,
            self.name,
            self.tag,
            self.non_linear_tag,
        )

    @property
    def file_param_names(self) -> str:
        return path.join(self.samples_path, "model.paramnames")

    @property
    def file_model_promises(self) -> str:
        return path.join(self.output_path, "model.promises")

    @property
    def file_model_info(self) -> str:
        return path.join(self.output_path, "model.info")

    @property
    def file_search_summary(self) -> str:
        return path.join(self.output_path, "search.summary")

    @property
    def file_results(self):
        return path.join(self.output_path, "model.results")

    @property
    @make_path
    def pdf_path(self) -> str:
        """
        The path to the directory in which images are stored.
        """
        return path.join(self.image_path, "pdf")

    @property
    @make_path
    def pickle_path(self) -> str:
        return path.join(self.make_path(), "pickles")

    def make_search_pickle_path(self) -> str:
        """
        Returns the path at which the search pickle should be saved
        """
        return path.join(self.pickle_path, "search.pickle")

    def make_model_pickle_path(self):
        """
        Returns the path at which the model pickle should be saved
        """
        return path.join(self.pickle_path, "model.pickle")

    def make_samples_pickle_path(self) -> str:
        """
        Returns the path at which the search pickle should be saved
        """
        return path.join(self.pickle_path, "samples.pickle")

    @make_path
    def make_path(self) -> str:
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

    # TODO : These should all be moved to the mult_nest.py ,module in a MultiNestPaths class. I dont know how t do this.

    @property
    def file_summary(self) -> str:
        return path.join(self.samples_path, "multinestsummary.txt")

    @property
    def file_weighted_samples(self):
        return path.join(self.samples_path, "multinest.txt")

    @property
    def file_phys_live(self) -> str:
        return path.join(self.samples_path, "multinestphys_live.points")

    @property
    def file_resume(self) -> str:
        return path.join(self.samples_path, "multinestresume.dat")

    def copy_from_sym(self):
        """
        Copy files from the sym-linked search folder to the samples folder.
        """

        src_files = os.listdir(self.path)
        for file_name in src_files:
            full_file_name = path.join(self.path, file_name)
            if path.isfile(full_file_name):
                shutil.copy(full_file_name, self.samples_path)

    def zip_remove(self):
        """
        Copy files from the sym linked search folder then remove the sym linked folder.
        """

        self.zip()

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

    def zip(self):

        try:
            with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as f:
                for root, dirs, files in os.walk(self.output_path):

                    for file in files:

                        # TODO : I removed lstrip("/") here, I think it is ok...


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
