from pathlib import Path
from typing import List, cast

from .import_ import Import, LineItem
from .item import Item, DirectoryItem


class File(DirectoryItem):
    def __init__(
            self,
            path: Path,
            prefix: str,
            parent: Item
    ):
        """
        A file to be edenised

        Parameters
        ----------
        path
            The path to the file prior to edenisation
        prefix
            A prefix to be prepended to package and module names
        """
        super().__init__(
            prefix,
            parent=parent
        )
        self._path = path

    def generate_target(self, output_path):
        with open(output_path / self.target_path, "w+") as f:
            for line in self.lines():
                f.write(f"{line.target_string}\n")

    @property
    def target_name(self) -> str:
        """
        The name of this file after edenisation
        """
        if not self.should_rename_modules or self.name == "__init__":
            return self.name
        return super().target_name

    @property
    def target_file_name(self) -> str:
        return f"{self.target_name}.py"

    @property
    def target_import_string(self) -> str:
        """
        The string for importing this file after edenisation
        """
        string = self.target_name.replace(".py", "")
        if self.should_rename_modules:
            string = f"{string} as {self.name}"
        return string

    @property
    def name(self) -> str:
        """
        The name of the file without the .py suffix
        """
        return super().name.replace(".py", "")

    @property
    def children(self) -> List[Import]:
        """
        Imports in the file
        """
        return self.imports

    @property
    def path(self) -> Path:
        """
        The path to the file
        """
        return self._path

    def lines(self):
        """
        Yield objects comprising 'lines'

        Open parentheses are used to determine if a 'line' goes across several
        true lines
        """
        with open(self.path) as f:
            line_item = None
            for line in f.read().split("\n"):
                new_item = LineItem(
                    line,
                    self
                )
                if line_item is None:
                    line_item = new_item
                else:
                    line_item += new_item

                if line_item.is_complete:
                    yield line_item
                    line_item = None

    @property
    def imports(self) -> List[Import]:
        """
        Imports in the file
        """
        return cast(
            List[Import],
            list(filter(
                lambda item: isinstance(
                    item, Import
                ),
                self.lines()
            ))
        )

    @property
    def project_imports(self) -> List[Import]:
        """
        Imports in the file that belong to the project
        """
        return [
            import_
            for import_
            in self.imports
            if import_.is_in_project
        ]
