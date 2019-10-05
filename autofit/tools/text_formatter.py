class TextFormatter:
    def __init__(self, line_length=90, indent=4):
        self.dict = dict()
        self.line_length = line_length
        self.indent = indent

    def add_to_dict(self, path_item_tuple: tuple, info_dict: dict):
        path_tuple = path_item_tuple[0]
        key = path_tuple[0]
        if len(path_tuple) == 1:
            info_dict[key] = path_item_tuple[1]
        else:
            if key not in info_dict:
                info_dict[key] = dict()
            self.add_to_dict(
                (path_item_tuple[0][1:], path_item_tuple[1]), info_dict[key]
            )

    def add(self, path_item_tuple: tuple):
        self.add_to_dict(path_item_tuple, self.dict)

    def dict_to_list(self, info_dict, line_length):
        lines = []
        for key, value in info_dict.items():
            indent_string = self.indent * " "
            if isinstance(value, dict):
                sub_lines = self.dict_to_list(
                    value, line_length=line_length - self.indent
                )
                lines.append(key)
                for line in sub_lines:
                    lines.append(f"{indent_string}{line}")
            else:
                value_string = str(value)
                space_string = max((line_length - len(key)), 1) * " "
                lines.append(f"{key}{space_string}{value_string}")
        return lines

    @property
    def list(self):
        return self.dict_to_list(self.dict, line_length=self.line_length)

    @property
    def text(self):
        return "\n".join(self.list)
