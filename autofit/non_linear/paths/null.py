from typing import Optional

from .abstract import AbstractPaths


class NullPaths(AbstractPaths):
    """
    Null version of paths object for avoiding writing of files to disk
    """

    def __init__(self):
        super().__init__()
        self.objects = dict()

    @AbstractPaths.parent.setter
    def parent(self, parent):
        pass

    @property
    def is_grid_search(self) -> bool:
        return False

    def create_child(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            is_identifier_in_paths: Optional[bool] = None,
            identifier: Optional[str] = None
    ) -> "AbstractPaths":
        return NullPaths()

    def save_named_instance(self, name: str, instance):
        pass

    def save_object(self, name: str, obj: object):
        self.objects[name] = obj

    def load_object(self, name: str):
        return self.objects[name]

    def remove_object(self, name: str):
        pass

    def is_object(self, name: str) -> bool:
        pass

    @property
    def is_complete(self) -> bool:
        return False

    def completed(self):
        pass

    def save_all(self, search_config_dict, info, pickle_files):
        pass

    def load_samples(self):
        pass

    def save_samples(self, samples):
        pass

    def load_samples_info(self):
        pass
