import os
import shutil

from autofit.aggregator import Aggregator


def update_identifiers(
        output_directory
):
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
