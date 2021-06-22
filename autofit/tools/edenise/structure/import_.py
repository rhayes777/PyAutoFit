import re
from pathlib import Path
from typing import Optional

from autofit.tools.edenise.structure.item import Item


class LineItem(Item):
    def __init__(
            self,
            string: str,
            parent: Item
    ):
        self.string = string
        super().__init__(
            parent=parent
        )

    def __new__(cls, string, parent):
        if string.startswith(
                "from"
        ) or string.startswith(
            "import"
        ):
            return object.__new__(Import)
        return object.__new__(LineItem)

    @property
    def open_count(self):
        return len(re.findall(r"\(", self.string))

    @property
    def close_count(self):
        return len(re.findall(r"\)", self.string))

    @property
    def is_open(self):
        return self.open_count > self.close_count

    def __add__(self, other):
        return LineItem(
            self.string + " " + other.string,
            self.parent
        )

    @property
    def children(self):
        """
        Imports don't have any children
        """
        return []

    @property
    def path(self) -> Path:
        return self.parent.path / self.string

    @property
    def name(self) -> str:
        return self.string

    @property
    def target_name(self) -> str:
        return self.string


class Import(LineItem):
    def __init__(
            self,
            string: str,
            parent: Optional[Item]
    ):
        """
        An import statement in a file

        Parameters
        ----------
        string
            The original line describing the import
        """
        match = re.match(
            r"from (\.+)([a-zA-Z0-9_.]*) import (.*)",
            string
        )
        if match is not None:
            level = parent
            for _ in match[1]:
                level = level.parent

            import_path = level.import_path
            if match[2] != "":
                import_path = f"{import_path}.{match[2]}"

            string = f"from {import_path} import {match[3]}"

        super().__init__(
            string=string,
            parent=parent,
        )

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
    def _parts(self):
        return self.string.split(" ")

    @property
    def items(self):
        strings = self.string.split("import ")[-1].split(", ")

        items = list()
        module = self.module

        for string in strings:
            if " as" in string:
                string, alias = string.split(" as ")
                item = As(module[string], alias)
            else:
                item = module[string]

            items.append(item)
        return items

    @property
    def module_string(self):
        return self._parts[1]

    @property
    def module_path(self):
        return self.module_string.split(".")

    @property
    def module(self):
        item = self.top_level
        for name in self.module_path[1:]:
            item = item[name]
        return item

    @property
    def target_import_string(self) -> str:
        """
        The string that will describe this import after edenisation
        """
        module_string = ".".join(map(
            self._edenise_string,
            self.module_path
        ))

        item_string = ", ".join([
            item.target_name
            for item
            in self.items
        ])

        return f"from {module_string} import {item_string}"

    @property
    def suffix(self) -> str:
        """
        The final component of this import.

        e.g. from os import path -> path
        """
        return self.string.split(" ")[-1]

    @property
    def path(self) -> Path:
        """
        The path to the file containing this import
        """
        import_string = self.string
        suffix = self.suffix
        if "*" in import_string:
            import_string = import_string.replace(
                " import *", ""
            ).replace(
                "from ", "import "
            )
            suffix = import_string.replace(
                "import ", ""
            )

        loc = {}
        try:
            exec(
                f"""
{import_string}
import inspect

is_class = inspect.isclass({suffix})

if is_class:
    path = inspect.getfile({suffix})
else:
    path = {suffix}.__file__
""",
                globals(),
                loc
            )
        except AttributeError as e:
            print(e)
        except (ModuleNotFoundError, SyntaxError) as e:
            raise e
        return Path(loc["path"])


class As:
    def __init__(self, item, alias):
        self.item = item
        self.alias = alias

    @property
    def target_name(self):
        return f"{self.item.target_name} as {self.alias}"
