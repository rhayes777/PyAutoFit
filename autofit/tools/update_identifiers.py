import logging
import os
import shutil
from hashlib import md5
from pathlib import Path
from typing import Union

import yaml
from autofit.database.sqlalchemy_ import sa

from autofit.aggregator import Aggregator as ClassicAggregator
from autofit.database.aggregator import Aggregator as DatabaseAggregator
from autofit.non_linear.paths.database import DatabasePaths
from autofit.tools.util import zip_directory

logger = logging.getLogger(
    __name__
)


def update_identifiers_from_file(
        output_directory: Union[str, os.PathLike],
        map_filename: str
):
    """
    Update identifiers in a given directory by loading and modifying their
    identifier files.

    This is necessary when a change to source code means that pickles can no
    longer be loaded. Searches must also be re-run to replace pickles.

    Parameters
    ----------
    output_directory
        A directory containing output results
    map_filename
        Filename of a file mapping old fields to new
    """
    with open(map_filename) as f:
        field_map = yaml.safe_load(f)

    update_identifiers_from_dict(
        output_directory,
        field_map
    )


def update_identifiers_from_dict(
        output_directory: Union[str, os.PathLike],
        field_map: dict
):
    """
    Update identifiers in a given directory by loading and modifying their
    identifier files.

    This is necessary when a change to source code means that pickles can no
    longer be loaded. Searches must also be re-run to replace pickles.

    Parameters
    ----------
    output_directory
        A directory containing output results
    field_map
        A dictionary mapping old fields to new
    """
    aggregator = ClassicAggregator(
        output_directory
    )
    for output in aggregator:
        directory = output.directory

        def remove_source():
            shutil.rmtree(
                directory,
                ignore_errors=True
            )

        print(f"Processing {directory}")
        identifier_filename = f"{directory}/.identifier"
        with open(identifier_filename) as f:
            hash_list = f.read().split("\n")

        for i in range(len(hash_list)):
            string = hash_list[i]
            if string in field_map:
                new_value = field_map[string]
                print(f"Replacing {string} with {new_value}")
                hash_list[i] = new_value

        new_identifier = md5(".".join(
            hash_list
        ).encode("utf-8")).hexdigest()


        old_identifier = os.path.split(identifier_filename)[0]
        old_identifier = os.path.split(old_identifier)[-1]

     #   if new_identifier == output.search.paths.identifier:
        if new_identifier == old_identifier:
            logger.warning(
                f"Skipping {directory} as there is no change"
            )

            zip_file = f"{os.path.split(identifier_filename)[0]}.zip"

            if os.path.exists(zip_file):
                remove_source()
            continue

        new_directory = Path(
            directory
        ).parent / new_identifier

        print(
            f"Moving output from {directory} to {new_directory}"
        )

        os.makedirs(
            new_directory,
            exist_ok=True
        )

        for file in os.listdir(
                directory
        ):
            if file.endswith(
                    ".pickle"
            ):
                print(f"Skipping {file}")
                continue
            if not os.path.exists(
                    new_directory / file
            ):
                shutil.move(
                    f"{directory}/{file}",
                    new_directory
                )

        with open(
            new_directory / ".identifier",
            "w+"
        ) as f:
            f.write(
                "\n".join(
                    hash_list
                )
            )

        zip_directory(
            new_directory
        )

        shutil.rmtree(
            new_directory
        )
        remove_source()
        try:
            os.remove(
                f"{directory}.zip"
            )
        except FileNotFoundError:
            pass


def update_directory_identifiers(
        output_directory: Union[str, os.PathLike]
):
    """
    Update identifiers in a given directory.

    When identifiers were computed through an out of date method this
    can be used to move the data to a new directory with the correct
    identifier.

    search.pickle is replaced to ensure its internal identifier matches
    that of the directory.

    Parameters
    ----------
    output_directory
        A directory containing output results
    """
    aggregator = ClassicAggregator(
        output_directory
    )
    for output in aggregator:
        paths = output.search.paths
        try:
            os.remove(
                f"{paths.output_path}.zip"
            )
        except FileNotFoundError:
            pass

        source_directory = output.directory
        paths._identifier = None
        paths.save_identifier()
        target_directory = paths.output_path

        logger.info(
            f"Moving output from {source_directory} to {target_directory}"
        )

        os.makedirs(
            target_directory,
            exist_ok=True
        )

        for file in os.listdir(
                source_directory
        ):
            target_filename = f"{target_directory}/{file}"
            if not os.path.exists(
                    target_filename
            ):
                shutil.move(
                    f"{source_directory}/{file}",
                    target_filename
                )

        paths.save_object("search", output.search)

        shutil.rmtree(
            source_directory
        )

        paths.zip_remove()


def update_database_identifiers(
        session: sa.orm.Session
):
    """
    Update identifiers for a database.

    Parameters
    ----------
    session
        A SQLAlchemy session connected to the database
    """
    aggregator = DatabaseAggregator(session)

    args = list()

    for output in aggregator:
        search = output["search"]
        model = output["model"]
        paths = DatabasePaths(
            session=session,
            name=output.name,
            path_prefix=output.path_prefix,
            unique_tag=output.unique_tag,
        )
        paths.search = search
        paths.model = model

        args.append({
            "old_id": output.id,
            "new_id": paths.identifier
        })

    session.execute(
        "UPDATE fit SET id = :new_id WHERE id = :old_id",
        args
    )
