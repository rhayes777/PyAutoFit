import logging
import os
import pickle
from pathlib import Path

from sqlalchemy.orm import Session

from .. import model as m
from ...mapper.model_object import Identifier

logger = logging.getLogger(
    __name__
)


class Scraper:
    def __init__(
            self,
            directory: str,
            session: Session
    ):
        self.directory = directory
        self.session = session

    def scrape(self):
        for fit in self._fits():
            self.session.add(
                fit
            )
        for grid_search in self._grid_searches():
            self.session.add(
                grid_search
            )

    def _fits(self):
        """
        Scrape data output into a directory tree so it can be added to the
        aggregator database.

        Returns
        -------
        Generator yielding Fit database objects
        """
        logger.info(
            f"Scraping directory {self.directory}"
        )
        from autofit.aggregator.aggregator import Aggregator as ClassicAggregator
        aggregator = ClassicAggregator(
            self.directory
        )
        logger.info(
            f"{len(aggregator)} searches found"
        )
        for item in aggregator:
            is_complete = os.path.exists(
                f"{item.directory}/.completed"
            )

            model = item.model
            samples = item.samples

            try:
                instance = samples.max_log_likelihood_instance
            except (AttributeError, NotImplementedError):
                instance = None

            fit = m.Fit(
                id=_make_identifier(
                    item
                ),
                name=item.search.name,
                unique_tag=item.search.unique_tag,
                model=model,
                instance=instance,
                is_complete=is_complete,
                info=item.info,
                max_log_likelihood=samples.max_log_likelihood_sample.log_likelihood
            )

            pickle_path = Path(item.pickle_path)
            _add_pickles(
                fit,
                pickle_path
            )

            yield fit

    def _grid_searches(
            self
    ):
        """
        Retrieve grid searches recursively from an output directory by
        searching for the .is_grid_search file.

        Should be called after adding Fits as it relies on querying fits

        Yields
        ------
        Fit objects representing grid searches with child fits associated
        """
        from autofit.aggregator.aggregator import Aggregator as ClassicAggregator
        for root, _, filenames in os.walk(self.directory):
            if ".is_grid_search" in filenames:
                path = Path(root)
                grid_search = m.Fit(
                    id=path.name,
                    is_grid_search=True
                )

                pickle_path = path / "pickles"
                _add_pickles(
                    grid_search,
                    pickle_path
                )

                aggregator = ClassicAggregator(
                    root
                )
                for item in aggregator:
                    fit = self._retrieve_model_fit(
                        item
                    )
                    grid_search.children.append(
                        fit
                    )
                yield grid_search

    def _retrieve_model_fit(
            self,
            item
    ):
        return self.session.query(
            m.Fit
        ).filter(
            m.Fit.id == _make_identifier(
                item
            )
        ).one()


def _make_identifier(
        item
):
    search = item.search
    model = item.model
    return str(Identifier([
        search,
        model,
        search.unique_tag
    ]))


def _add_pickles(
        fit,
        pickle_path
):
    for filename in os.listdir(
            pickle_path
    ):
        with open(
                pickle_path / filename,
                "r+b"
        ) as f:
            fit[
                filename.split(".")[0]
            ] = pickle.load(f)
