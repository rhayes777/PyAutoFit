import re
from pathlib import Path
from typing import Optional, List

from autofit.tools.edenise.structure.item import Item


class LineItem(Item):
    def __init__(
            self,
            string: str,
            parent: Item
    ):
        """
        A line in a file.

        Special consideration is given to lines containing imports.

        When parentheses are not balanced a line can span multiple lines in a file.

        Parameters
        ----------
        string
            The string on the line
        parent
            The file containing the line
        """
        self.string = string
        super().__init__(
            parent=parent
        )

    def __new__(cls, string, parent):
        if re.match(
                r" *(import|from).*",
                string
        ):
            return object.__new__(Import)
        return object.__new__(LineItem)

    @property
    def is_complete(self) -> bool:
        """
        Is this a complete line? If a doc string or parenthesis has been
        opened then it is not complete.

        Doc strings take precedence to mitigate issue with unbalanced
        parentheses in doc strings.
        """
        if self.is_open_doc_string:
            return False
        if not self.is_doc_string and self.is_open:
            return False
        return True

    @property
    def lines(self) -> List[str]:
        """
        Lines in this 'line' split by newline.

        Multiple lines occur when a doc string or parenthesis is opened.
        """
        return self.string.split("\n")

    @property
    def is_doc_string(self):
        """
        Does this line start with the opening of a doc string?
        """
        return '"""' in self.lines[0]

    @property
    def is_open_doc_string(self) -> bool:
        """
        Has a doc string been opened but not closed?
        """
        if self.is_doc_string:
            return len(self.lines) == 1 or '"""' not in self.lines[-1]
        return False

    @property
    def open_count(self) -> int:
        """
        How many opening parentheses have been encountered?
        """
        return len(re.findall(r"\(", self.string))

    @property
    def close_count(self):
        """
        How many closing parentheses have been encountered?
        """
        return len(re.findall(r"\)", self.string))

    @property
    def is_open(self):
        """
        Is there a parenthesis or quote imbalance?
        """
        return self.open_count > self.close_count

    def __add__(self, other):
        return LineItem(
            self.string + "\n" + other.string,
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
    def is_function(self):
        return re.match(r"def +\w+", self.string) is not None

    @property
    def target_string(self) -> str:
        if self.is_function and self.should_remove_type_annotations:
            result = list()
            open_square_count = 0
            open_bracket_count = 0

            should_add = True
            for i, character in enumerate(self.string):
                def is_window(string):
                    return self.string[i: i + len(string)] == string

                if is_window(" -> ") or is_window(" ->") or is_window("-> ") or is_window("->"):
                    should_add = False
                if character == ":":
                    should_add = open_bracket_count == 0
                if character == "=":
                    should_add = True
                if character == "(":
                    open_bracket_count += 1
                if character == ")":
                    open_bracket_count -= 1
                    if open_bracket_count == 0:
                        should_add = True
                if character == "[":
                    open_square_count += 1
                if character == "]":
                    open_square_count -= 1
                if character in (",", "\n") and open_square_count == 0:
                    should_add = True

                if should_add:
                    result.append(character)

            return "".join(result)
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
    def target_string(self) -> str:
        return self.target_import_string

    @property
    def _parts(self):
        return self.string.lstrip().split(" ")

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
        return self._parts[1].replace(",", "")

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
    def is_in_project(self) -> bool:
        """
        Is this object within the top level object?
        """
        string = self.string.lstrip()
        for dependency in self.eden_dependencies:
            if (string.startswith(
                    f"from {dependency}"
            ) or string.startswith(
                f"import {dependency}"
            )):
                return True
        return False

    @property
    def _space_prefix(self) -> str:
        """
        A string of spaces at the start of the import string
        """
        return re.findall(
            f"( *).*",
            self.string
        )[0]

    @property
    def target_import_string(self) -> str:
        """
        The string that will describe this import after edenisation
        """
        if not self.is_in_project:
            return self.string

        item = self.top_level
        module_string = item.target_file_name

        for name in self.module_path[1:]:
            item = item[name]
            module_string = f"{module_string}.{item.target_name}"

        item_string = ", ".join([
            f"{item.target_import_string}"
            for item
            in self.items
        ])

        return f"{self._space_prefix}from {module_string} import {item_string}"


class As:
    def __init__(self, item, alias):
        self.item = item
        self.alias = alias

    @property
    def target_import_string(self):
        return f"{self.item.target_import_string} as {self.alias}"
