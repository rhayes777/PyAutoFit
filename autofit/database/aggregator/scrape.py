import logging
from pathlib import Path
from typing import Optional, Union

from .. import model as m
from ..sqlalchemy_ import sa
from autofit.aggregator.search_output import SearchOutput

logger = logging.getLogger(__name__)


class Scraper:
    def __init__(
        self,
        directory: Union[Path, str],
        session: sa.orm.Session,
        reference: Optional[dict] = None,
    ):
        """
        Facilitates scraping of data output into a directory
        into the database.

        Parameters
        ----------
        directory
            A directory in which data has been stored
        session
            A database session
        reference
            A dictionary mapping search names to model paths
        """
        self.directory = directory
        self.session = session
        self.reference = reference

        from autofit.aggregator.aggregator import Aggregator as ClassicAggregator

        self.aggregator = ClassicAggregator.from_directory(
            self.directory,
            reference=self.reference,
        )

    def scrape(self):
        """
        Recursively scrape fits from the directory and
        add them to the session
        """
        for fit in self._fits():
            self.session.add(fit)
        for grid_search in self._grid_searches():
            self.session.add(grid_search)

    def _fits(self):
        """
        Scrape data output into a directory tree so it can be added to the
        aggregator database.

        Returns
        -------
        Generator yielding Fit database objects
        """
        logger.info(f"Scraping directory {self.directory}")
        logger.info(f"{len(self.aggregator)} searches found")
        for item in self.aggregator:
            logger.info(
                f"Creating fit for: "
                f"{item.path_prefix} "
                f"{item.unique_tag} "
                f"{item.name} "
                f"{item.id} "
            )

            try:
                fit = self._retrieve_model_fit(item)
                logger.warning(f"Fit already existed with identifier {item.id}")
            except sa.orm.exc.NoResultFound:
                fit = m.Fit(
                    id=item.id,
                    name=item.name,
                    unique_tag=item.unique_tag,
                    model=item.model,
                    instance=item.instance,
                    is_complete=item.is_complete,
                    info=item.info,
                    max_log_likelihood=item.max_log_likelihood,
                    parent_id=item.parent_identifier,
                )

            _add_files(fit, item)
            for i, child_analysis in enumerate(item.child_analyses):
                child_fit = m.Fit(
                    id=f"{item.id}_{i}",
                )
                _add_files(child_fit, child_analysis)
                fit.children.append(child_fit)

            yield fit

    def _grid_searches(self):
        """
        Retrieve grid searches recursively from an output directory by
        searching for the .is_grid_search file.

        Should be called after adding Fits as it relies on querying fits

        Yields
        ------
        Fit objects representing grid searches with child fits associated
        """
        for item in self.aggregator.grid_searches():
            grid_search = m.Fit(
                id=item.id,
                unique_tag=item.unique_tag,
                is_grid_search=True,
                is_complete=item.is_complete,
            )

            _add_files(grid_search, item)

            for search in item.children:
                fit = self._retrieve_model_fit(search)
                grid_search.children.append(fit)
            yield grid_search

    def _retrieve_model_fit(self, item) -> m.Fit:
        """
        Retrieve a Fit, if one exists, corresponding to a given SearchOutput

        Parameters
        ----------
        item
            A SearchOutput from the classic Aggregator

        Returns
        -------
        A fit with the corresponding identifier

        Raises
        ------
        NoResultFound
            If no fit is found with the identifier
        """
        return self.session.query(m.Fit).filter(m.Fit.id == item.id).one()


def _add_files(fit: m.Fit, item: SearchOutput):
    """
    Load files from the path and add them to the database.

    Parameters
    ----------
    fit
        A fit to which the pickles belong
    """
    try:
        fit.samples = item.samples
    except AttributeError:
        logger.warning(f"Failed to load samples for {fit.id}")

    for json_output in item.jsons:
        fit.set_json(json_output.name, json_output.dict)

    for pickle_output in item.pickles:
        fit.set_pickle(pickle_output.name, pickle_output.value)

    for array_output in item.arrays:
        try:
            fit.set_array(array_output.name, array_output.value)
        except ValueError:
            logger.debug(f"Failed to load array {array_output.name} for {fit.id}")

    for hdu_output in item.hdus:
        fit.set_hdu(hdu_output.name, hdu_output.value)
