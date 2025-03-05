import re
from enum import Enum


def subplot_filename(subplot: Enum) -> str:
    subplot_type = subplot.__class__
    return (
        re.sub(
            r"([A-Z])",
            r"_\1",
            subplot_type.__name__,
        )
        .lower()
        .lstrip("_")
    )
