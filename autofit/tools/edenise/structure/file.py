from pathlib import Path
from typing import List

from .import_ import Import
from .item import Item


class File(Item):
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

    @property
    def imports(self) -> List[Import]:
        """
        Imports in the file
        """
        imports = list()
        with open(self.path) as f:
            for line in f.read().split("\n"):
                if line.startswith(
                        "from"
                ) or line.startswith(
                    "import"
                ):
                    import_ = Import(
                        line,
                        parent=self
                    )
                    imports.append(
                        import_
                    )
        return imports

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
