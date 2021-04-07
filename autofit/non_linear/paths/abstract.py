import os
import re
from abc import ABC
from configparser import NoSectionError
from functools import wraps

from autoconf import conf
from autofit.non_linear.log import logger


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
