import re


def get_class_path(cls: type) -> str:
    """
    The full import path of the type
    """
    return re.search("'(.*)'", str(cls))[1]
