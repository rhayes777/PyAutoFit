import logging
import os
import pickle
from pathlib import Path
from typing import Optional, Union

from .. import model as m
from ..sqlalchemy_ import sa
from ...mapper.model_object import Identifier

logger = logging.getLogger(
    __name__
)


def _parent_identifier(
        directory: str
) -> Optional[str]:
    """
    Read the parent identifier for a fit in a directory.

    Defaults to None if no .parent_identifier file is found.
    """
    try:
        with open(f"{directory}/.parent_identifier") as f:
            return f.read()
    except FileNotFoundError:
        return None


class Scraper:
    def __init__(
            self,
            directory: Union[Path, str],
            session: sa.orm.Session
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
        """
        self.directory = directory
        self.session = session

    def scrape(self):
        """
        Recursively scrape fits from the directory and
        add them to the session
        """
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

            parent_identifier = _parent_identifier(
                directory=item.directory
            )

            model = item.model
            samples = item.samples

            try:
                instance = samples.max_log_likelihood_instance
            except (AttributeError, NotImplementedError):
                instance = None

            identifier = _make_identifier(item)

            logger.info(
                f"Creating fit for: "
                f"{item.search.paths.path_prefix} "
                f"{item.search.unique_tag} "
                f"{item.search.name} "
                f"{identifier} ")

            try:
                fit = self._retrieve_model_fit(
                    item
                )
                logger.warning(
                    f"Fit already existed with identifier {identifier}"
                )
            except sa.orm.exc.NoResultFound:
                try:
                    log_likelihood = samples.max_log_likelihood_sample.log_likelihood
                except AttributeError:
                    log_likelihood = None
                fit = m.Fit(
                    id=identifier,
                    name=item.search.name,
                    unique_tag=item.search.unique_tag,
                    model=model,
                    instance=instance,
                    is_complete=is_complete,
                    info=item.info,
                    max_log_likelihood=log_likelihood,
                    parent_id=parent_identifier
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

                is_complete = (path / ".completed").exists()

                with open(
                        path / ".is_grid_search"
                ) as f:
                    unique_tag = f.read()

                grid_search = m.Fit(
                    id=path.name,
                    unique_tag=unique_tag,
                    is_grid_search=True,
                    parent_id=_parent_identifier(
                        root
                    ),
                    is_complete=is_complete
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
    ) -> m.Fit:
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
        return self.session.query(
            m.Fit
        ).filter(
            m.Fit.id == _make_identifier(
                item
            )
        ).one()


def _make_identifier(
        item
) -> str:
    """
    Create a unique identifier for a SearchOutput.

    This accounts for the Search, Model and unique_tag

    Parameters
    ----------
    item
        An output from the classic aggregator

    Returns
    -------
    A unique identifier that is sensitive to changes that affect
    the search
    """
    search = item.search
    model = item.model
    return str(Identifier([
        search,
        model,
        search.unique_tag
    ]))


def _add_pickles(
        fit: m.Fit,
        pickle_path: Path
):
    """
    Load pickles from the path and add them to the database.

    Parameters
    ----------
    fit
        A fit to which the pickles belong
    pickle_path
        The path in which the pickles are stored
    """
    try:
        filenames = os.listdir(
            pickle_path
        )
    except FileNotFoundError as e:
        logger.exception(e)
        filenames = []

    for filename in filenames:

        try:
            with open(
                    pickle_path / filename,
                    "r+b"
            ) as f:
                fit[
                    filename.split(".")[0]
                ] = pickle.load(f)
        except (pickle.UnpicklingError, ModuleNotFoundError) as e:

            if filename == "dynesty.pickle":
                continue

            raise pickle.UnpicklingError(f"Failed to unpickle: {pickle_path} {filename}") from e
