import os
from pathlib import Path
from typing import List, Optional

from .file import File
from .item import DirectoryItem


class Package(DirectoryItem):
    def __init__(
            self,
            path: Path,
            prefix: str,
            is_top_level: bool,
            parent: Optional["Package"] = None,
            eden_dependencies: Optional[List[str]] = None,
            should_rename_modules: bool = False,
            should_remove_type_annotations: bool = False
    ):
        """
        A package in the project.

        Parameters
        ----------
        path
            The path to the package before edenisation
        prefix
            A prefix that must be prepended to all packages and modules
        is_top_level
            Is this the top level package of the project?
        eden_dependencies
            Other projects for which imports should be converted
        """
        super().__init__(
            prefix,
            parent=parent
        )
        self._path = path

        self.is_top_level = is_top_level
        self._eden_dependencies = eden_dependencies or list()
        self._should_rename_modules = should_rename_modules
        self._should_remove_type_annotations = should_remove_type_annotations

    @property
    def should_rename_modules(self):
        return self._should_rename_modules

    @property
    def should_remove_type_annotations(self):
        return self._should_remove_type_annotations

    @property
    def eden_dependencies(self):
        return self._eden_dependencies + [self.name]

    def generate_target(self, output_path: Path):
        (output_path / self.target_path).mkdir(
            parents=True,
            exist_ok=True
        )
        for child in self.children:
            child.generate_target(
                output_path
            )

    @property
    def target_file_name(self):
        return self.target_name

    @property
    def children(self) -> List[DirectoryItem]:
        """
        Packages and files contained directly in this package
        """
        children = list()
        for item in os.listdir(
                self.path
        ):
            item_path = self.path / item
            if item.endswith(".py"):
                children.append(
                    File(
                        item_path,
                        self.prefix,
                        parent=self
                    )
                )

            if os.path.isdir(item_path):
                if "__init__.py" in os.listdir(
                        item_path
                ):
                    children.append(
                        Package(
                            item_path,
                            prefix=self.prefix,
                            is_top_level=False,
                            parent=self
                        )
                    )
        return children

    @property
    def target_path(self) -> Path:
        """
        The path this object will have after edenisation
        """
        target_path = super().target_path
        #       target_path = Path(str(target_path).replace("Auto", ""))
        #       print(target_path)
        #       stop
        if self.is_top_level:
            target_path = Path(
                self.prefix
            ) / self.target_name / "python" / target_path
        return target_path

    @property
    def path(self) -> Path:
        """
        The path to this package prior to edenisation
        """
        return self._path
