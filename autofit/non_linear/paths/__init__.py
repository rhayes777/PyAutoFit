import logging
from pathlib import Path
from typing import Optional

from .abstract import AbstractPaths
from .database import DatabasePaths
from .directory import DirectoryPaths
from .null import NullPaths

logger = logging.getLogger(__name__)


def paths_from(
    name: Optional[str] = None,
    path_prefix: Optional[Path] = None,
    unique_tag: Optional[str] = None,
    session=None,
    paths: Optional[AbstractPaths] = None,
    save_all_samples: bool = False,
) -> AbstractPaths:
    """
    The output paths of a search: ``paths`` when given; otherwise ``DatabasePaths``
    when a database ``session`` is given, ``DirectoryPaths`` when the search has a
    ``name`` or ``path_prefix``, and ``NullPaths`` (no output) when it has neither.
    """
    if paths:
        return paths
    if session is not None:
        logger.debug("Session found. Using database.")
        return DatabasePaths(
            name=name,
            path_prefix=path_prefix,
            session=session,
            save_all_samples=save_all_samples,
            unique_tag=unique_tag,
        )
    if name is not None or path_prefix:
        logger.debug("Session not found. Using directory output.")
        return DirectoryPaths(name=name, path_prefix=path_prefix, unique_tag=unique_tag)
    return NullPaths()
