import shutil
from pathlib import Path

from .file import File
from .import_ import Import
from .item import Item
from .line import LineItem
from .package import Package


def edenise(
        root_directory,
        name,
        prefix,
        eden_prefix,
        eden_dependencies,
        target_eden_directory=None,
        should_rename_modules=False,
        should_remove_type_annotations=False
):
    target_directory = f"{root_directory}/../eden/{name}_eden"

    print(f"Creating {target_directory}...")
    shutil.rmtree(
        target_directory,
        ignore_errors=True
    )
    shutil.copytree(
        root_directory,
        target_directory,
        symlinks=True
    )

    target_directory = Path(target_directory)

    package = Package(
        target_directory / name,
        prefix=eden_prefix,
        is_top_level=True,
        eden_dependencies=eden_dependencies,
        should_rename_modules=should_rename_modules,
        should_remove_type_annotations=should_remove_type_annotations
    )

    target_eden_directory = Path(target_eden_directory or target_directory)

    package.generate_target(
        target_eden_directory
    )
