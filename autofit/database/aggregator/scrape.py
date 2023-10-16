import csv
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional, Union, Generator, Tuple

import numpy as np

from .. import model as m
from ..sqlalchemy_ import sa
from autofit.non_linear.samples.pdf import SamplesPDF
from autofit.non_linear.samples.sample import samples_from_iterator

logger = logging.getLogger(__name__)


def _parent_identifier(directory: str) -> Optional[str]:
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
        from autofit.aggregator.aggregator import Aggregator as ClassicAggregator

        aggregator = ClassicAggregator.from_directory(
            self.directory,
            reference=self.reference,
        )
        logger.info(f"{len(aggregator)} searches found")
        for item in aggregator:
            parent_identifier = _parent_identifier(directory=item.directory)

            model = item.model
            samples = item.samples

            logger.info(
                f"Creating fit for: "
                f"{item.search.paths.path_prefix} "
                f"{item.search.unique_tag} "
                f"{item.search.name} "
                f"{item.id} "
            )

            try:
                instance = samples.max_log_likelihood()
            except (AttributeError, NotImplementedError):
                instance = None

            try:
                fit = self._retrieve_model_fit(item)
                logger.warning(f"Fit already existed with identifier {item.id}")
            except sa.orm.exc.NoResultFound:
                try:
                    log_likelihood = samples.max_log_likelihood_sample.log_likelihood
                except AttributeError:
                    log_likelihood = None
                fit = m.Fit(
                    id=item.id,
                    name=item.search.name,
                    unique_tag=item.search.unique_tag,
                    model=model,
                    instance=instance,
                    is_complete=item.is_complete,
                    info=item.info,
                    max_log_likelihood=log_likelihood,
                    parent_id=parent_identifier,
                )

            _add_files(fit, Path(item.files_path))
            for i, child_analysis in enumerate(item.child_analyses):
                child_fit = m.Fit(
                    id=f"{item.id}_{i}",
                )
                _add_files(child_fit, child_analysis.files_path)
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
        from autofit.aggregator.aggregator import Aggregator as ClassicAggregator

        for root, _, filenames in os.walk(self.directory):
            if ".is_grid_search" in filenames:
                path = Path(root)

                is_complete = (path / ".completed").exists()

                with open(path / ".is_grid_search") as f:
                    unique_tag = f.read()

                grid_search = m.Fit(
                    id=path.name,
                    unique_tag=unique_tag,
                    is_grid_search=True,
                    parent_id=_parent_identifier(root),
                    is_complete=is_complete,
                )

                _add_files(grid_search, path / "files")

                aggregator = ClassicAggregator.from_directory(root)
                for item in aggregator:
                    fit = self._retrieve_model_fit(item)
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


def names_and_paths(
    files_path: Path,
    suffix: str,
) -> Generator[Tuple[str, Path], None, None]:
    """
    Get the names and paths of files with a given suffix in a directory.

    Parameters
    ----------
    files_path
        The path in which the files are stored
    suffix
        The suffix of the files to retrieve (e.g. ".json")

    Returns
    -------
    A generator of tuples of the form (name, path) where name is the path to the file
    joined by . without the suffix and path is the path to the file
    """
    for file in list(files_path.rglob(f"*{suffix}")):
        name = ".".join(file.relative_to(files_path).with_suffix("").parts)
        yield name, file


def _add_files(fit: m.Fit, files_path: Path):
    """
    Load files from the path and add them to the database.

    Parameters
    ----------
    fit
        A fit to which the pickles belong
    files_path
        The path in which the JSONs are stored
    """
    info_path = files_path / "samples_info.json"
    samples_path = files_path / "samples.csv"
    if info_path.exists() and samples_path.exists():
        with open(info_path) as f:
            info_json = json.load(f)
        with open(samples_path) as f:
            sample_list = samples_from_iterator(csv.reader(f))

        fit.samples = SamplesPDF.from_list_info_and_model(
            sample_list=sample_list,
            samples_info=info_json,
            model=fit.model,
        )

    for name, path in names_and_paths(files_path, ".json"):
        with open(path) as f:
            fit.set_json(name, json.load(f))
    for name, path in names_and_paths(files_path, ".csv"):
        try:
            with open(path) as f:
                fit.set_array(name, np.loadtxt(f, delimiter=","))
        except ValueError:
            logger.debug(f"Failed to load array from {path}")

    for name, path in names_and_paths(files_path, ".pickle"):
        try:
            with open(path, "r+b") as f:
                fit[name] = pickle.load(f)
        except (pickle.UnpicklingError, ModuleNotFoundError) as e:
            if path.name == "dynesty.pickle":
                continue

            raise pickle.UnpicklingError(f"Failed to unpickle: {path}") from e

    for name, path in names_and_paths(files_path, ".fits"):
        from astropy.io import fits

        with open(path, "rb") as f:
            fit.set_hdu(name, fits.PrimaryHDU.readfrom(f))
