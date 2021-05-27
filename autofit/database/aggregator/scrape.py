import os

from .. import model as m
from ...mapper.model_object import Identifier


def scrape_directory(directory: str):
    """
    Scrape data output into a directory tree so it can be added to the
    aggregator database.

    Parameters
    ----------
    directory
        A directory containing the output from previous fits

    Returns
    -------
    Generator yielding Fit database objects
    """
    from autofit.aggregator.aggregator import Aggregator as ClassicAggregator
    aggregator = ClassicAggregator(
        directory
    )
    for item in aggregator:
        is_complete = os.path.exists(
            f"{item.directory}/.completed"
        )

        model = item.model
        samples = item.samples
        search = item.search

        try:
            instance = samples.max_log_likelihood_instance
        except (AttributeError, NotImplementedError):
            instance = None

        fit = m.Fit(
            id=str(Identifier([
                search,
                model,
                search.unique_tag
            ])),
            model=model,
            instance=instance,
            is_complete=is_complete,
            info=item.info
        )

        pickle_path = item.pickle_path
        for pickle_name in os.listdir(
                pickle_path
        ):
            with open(
                    os.path.join(
                        pickle_path,
                        pickle_name
                    ),
                    "r+b"
            ) as f:
                fit[pickle_name.replace(
                    ".pickle",
                    ""
                )] = f.read()
        yield fit
