import shutil
from configparser import ConfigParser
from os import walk
from pathlib import Path

from .converter import Converter, Line
from .structure import Import, Item, File, Package


def edenise_directory(
        root_directory
):
    try:
        config = ConfigParser()
        config.read(
            f"{root_directory}/eden.ini"
        )

        edenise(
            root_directory,
            config.get("eden", "name"),
            config.get("eden", "prefix")
        )
    except ValueError:
        print("Usage: ./edenise.py root_directory")
        exit(1)


def edenise(
        root_directory,
        name,
        prefix
):
    target_directory = f"{root_directory}/../eden/{name}_eden"

    print(f"Creating {target_directory}...")
    shutil.copytree(
        root_directory,
        target_directory,
        symlinks=True
    )

    converter = Converter.from_prefix_and_source_directory(
        name=name,
        prefix=prefix,
        source_directory=f"{root_directory}/{name}"
    )

    for root, _, files in walk(f"{target_directory}/test_{name}"):
        for file in files:
            if file.endswith(".py"):
                with open(f"{root}/{file}", "r+") as f:
                    string = f.read()
                    f.seek(0)
                    f.write(
                        converter.convert(
                            string
                        )
                    )
                    f.truncate()

    for root, _, files in walk(f"{target_directory}/{name}"):
        try:
            for file in ["mock.py", "mock_real.py"]:
                with open(f"{root}/{file}", "r+") as f:
                    string = f.read()
                    f.seek(0)
                    f.write(
                        converter.convert(
                            string
                        )
                    )
                    f.truncate()
        except FileNotFoundError:
            continue

    open(f"{target_directory}/{name}/__init__.py", "w+").close()

    target_directory = Path(target_directory)

    package = Package(
        target_directory / name,
        prefix="VIS_CTI",
        is_top_level=True
    )
    package.generate_target(
        target_directory
    )
