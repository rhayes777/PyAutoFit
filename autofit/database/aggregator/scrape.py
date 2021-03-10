import os

from autofit.aggregator.aggregator import Aggregator as ClassicAggregator
from .. import model as m


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
    aggregator = ClassicAggregator(
        directory
    )
    for item in aggregator:
        model = item.model
        samples = item.samples
        instance = samples.max_log_likelihood_instance
        fit = m.Fit(
            model=model,
            instance=instance,
            phase_name=item.name
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
