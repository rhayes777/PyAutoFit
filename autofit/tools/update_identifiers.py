import os
import shutil

from autofit.aggregator import Aggregator


def update_identifiers(
        output_directory: str
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
    aggregator = Aggregator(
        output_directory
    )
    for output in aggregator:
        paths = output.search.paths
        source_directory = paths.output_path
        paths._identifier = None
        target_directory = paths.output_path

        for file in os.listdir(
                source_directory
        ):
            if not os.path.exists(
                    f"{target_directory}/{file}"
            ):
                shutil.move(
                    f"{source_directory}/{file}",
                    target_directory
                )
        shutil.rmtree(
            source_directory
        )

        paths.save_object("search", output.search)
