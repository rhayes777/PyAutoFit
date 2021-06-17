from pathlib import Path

from autofit.tools.edenise.structure.item import Item


class Import(Item):
    def __init__(self, string: str):
        """
        An import statement in a file

        Parameters
        ----------
        string
            The original line describing the import
        """
        super().__init__()
        self.string = string

        self.loc = {}
        exec(
            f"""
{self.string}
import inspect

is_class = inspect.isclass({self.suffix})

if is_class:
    path = inspect.getfile({self.suffix})
else:
    path = {self.suffix}.__file__
""",
            globals(),
            self.loc
        )

    @property
    def children(self):
        """
        Imports don't have any children
        """
        return []

    @property
    def file(self) -> Item:
        """
        The file containing this import
        """
        path = str(self.path)

        def get_from_item(
                item
        ):
            for child in item.children:
                if path == str(child.path):
                    return child
                if path.startswith(
                        str(child.path)
                ):
                    return get_from_item(
                        child
                    )

        return get_from_item(
            self.top_level
        )

    @property
    def target_import_string(self) -> str:
        """
        The string that will describe this import after edenisation
        """
        parts = self.string.split(" ")

        module = parts[1]
        items = parts[3:]

        module_string = ".".join(map(
            self._edenise_string,
            module.split(".")
        ))

        item_string = " ".join(
            items
        )

        return f"from {module_string} import {item_string}"

    @property
    def suffix(self) -> str:
        """
        The final component of this import.

        e.g. from os import path -> path
        """
        return self.string.split(" ")[-1]

    # def is_class(self) -> bool:
    #     return self.loc["is_class"]

    @property
    def path(self) -> Path:
        """
        The path to the file containing this import
        """
        return Path(self.loc["path"])
