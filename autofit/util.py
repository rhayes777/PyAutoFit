import builtins
import importlib
import re
from typing import List, Type


def get_class_path(cls: type) -> str:
    """
    The full import path of the type
    """
    return re.search("'(.*)'", str(cls))[1]


def get_class(class_path: str) -> Type[object]:
    return GetClass(
        class_path
    ).cls


class GetClass:
    def __init__(self, class_path):
        self.class_path = class_path

    @property
    def _class_path_array(self) -> List[str]:
        """
        A list of strings describing the module and class of the
        real object represented here
        """
        return self.class_path.split(".")

    @property
    def _class_name(self) -> str:
        """
        The name of the real class
        """
        return self._class_path_array[-1]

    @property
    def _module_path(self) -> str:
        """
        The path of the module containing the real class
        """
        return ".".join(self._class_path_array[:-1])

    @property
    def _module(self):
        """
        The module containing the real class
        """
        try:
            return importlib.import_module(
                self._module_path
            )
        except ValueError:
            return builtins

    @property
    def cls(self) -> Type[object]:
        """
        The class of the real object
        """
        return getattr(
            self._module,
            self._class_name
        )
