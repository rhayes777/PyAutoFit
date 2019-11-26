import os
from functools import wraps

from autofit import conf
from autofit.mapper import link


def make_path(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        full_path = func(*args, **kwargs)
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except FileExistsError:
                pass
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
            first_arg = kwargs.pop("phase_name", None)

        func(
            self,
            paths=Paths(
                phase_name=first_arg,
                phase_tag=kwargs.pop("phase_tag", None),
                phase_folders=kwargs.pop("phase_folders", tuple()),
                phase_path=kwargs.pop("phase_path", None),
            ),
            **kwargs
        )

    return wrapper


class Paths:
    def __init__(
        self, phase_name="", phase_tag=None, phase_folders=tuple(), phase_path=None
    ):
        if not isinstance(phase_name, str):
            raise ValueError("Phase name must be a string")
        self.phase_path = phase_path or "/".join(phase_folders)
        self.phase_name = phase_name
        self.phase_tag = phase_tag or ""

    @property
    def path(self):
        return link.make_linked_folder(self.sym_path)

    def __eq__(self, other):
        return isinstance(other, Paths) and all(
            [
                self.phase_path == other.phase_path,
                self.phase_name == other.phase_name,
                self.phase_tag == other.phase_tag,
            ]
        )

    @property
    def phase_folders(self):
        return self.phase_path.split("/")

    @property
    def backup_path(self) -> str:
        """
        The path to the backed up optimizer folder.
        """
        return "/".join(
            filter(
                lambda item: len(item) > 0,
                [
                    conf.instance.output_path,
                    self.phase_path,
                    self.phase_name,
                    self.phase_tag,
                    "optimizer_backup",
                ],
            )
        )

    @property
    @make_path
    def phase_output_path(self) -> str:
        """
        The path to the output information for a phase.
        """
        return "{}/{}/{}/{}/".format(
            conf.instance.output_path, self.phase_path, self.phase_name, self.phase_tag
        )

    @property
    def sym_path(self) -> str:
        return "{}/{}/{}/{}/optimizer".format(
            conf.instance.output_path, self.phase_path, self.phase_name, self.phase_tag
        )

    @property
    def file_param_names(self) -> str:
        return "{}/{}".format(self.path, "multinest.paramnames")

    @property
    def file_model_info(self) -> str:
        return "{}/{}".format(self.phase_output_path, "model.info")

    @property
    @make_path
    def image_path(self) -> str:
        """
        The path to the directory in which images are stored.
        """
        return "{}image/".format(self.phase_output_path)

    @property
    @make_path
    def pdf_path(self) -> str:
        """
        The path to the directory in which images are stored.
        """
        return "{}pdf/".format(self.image_path)

    def make_optimizer_pickle_path(self) -> str:
        """
        Create the path at which the optimizer pickle should be saved
        """
        return "{}/optimizer.pickle".format(self.make_path())

    def make_model_pickle_path(self):
        """
        Create the path at which the model pickle should be saved
        """
        return "{}/model.pickle".format(self.make_path())

    @make_path
    def make_path(self) -> str:
        """
        Create the path to the folder at which the metadata and optimizer pickle should
        be saved
        """
        return "{}/{}/{}/{}/".format(
            conf.instance.output_path, self.phase_path, self.phase_name, self.phase_tag
        )

    @property
    def file_summary(self) -> str:
        return "{}/{}".format(self.backup_path, "multinestsummary.txt")

    @property
    def file_weighted_samples(self):
        return "{}/{}".format(self.backup_path, "multinest.txt")

    @property
    def file_results(self):
        return "{}/{}".format(self.phase_output_path, "model.results")
