from typing import Optional, List

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from autofit.database import query as q
from .abstract import AbstractAggregator
from .scrape import scrape_directory
from .. import model as m
from ..query.query import AbstractQuery


class ListAggregator(AbstractAggregator):
    def __init__(self, fits):
        self._fits = fits

    @property
    def fits(self):
        return self._fits


class Aggregator(AbstractAggregator):
    def __init__(
            self,
            session: Session,
            filename: Optional[str] = None
    ):
        """
        Query results from an intermediary SQLite database.

        Results can be scraped from a directory structure and stored in the database.

        Parameters
        ----------
        session
            A session for communicating with the database.
        filename
        """
        self.session = session
        self.filename = filename
        self._fits = None

    @property
    def fits(self):
        if self._fits is None:
            self._fits = self._fits_for_query(
                "SELECT id FROM fit"
            )
        return self._fits

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.filename}>"

    def __getattr__(self, name):
        if name in m.fit_attributes:
            return q.A(name)
        return q.Q(name)

    def query(self, predicate: AbstractQuery) -> ListAggregator:
        # noinspection PyUnresolvedReferences
        """
        Apply a query on the model.

        Parameters
        ----------
        predicate
            A predicate constructed to express which models should be included.

        Returns
        -------
        A list of objects that match the predicate

        Examples
        --------
        >>>
        >>> aggregator = Aggregator.from_database(
        >>>     "my_database.sqlite"
        >>> )
        >>>
        >>> lens = aggregator.galaxies.lens
        >>>
        >>> aggregator.filter((lens.bulge == EllipticalCoreSersic) & (lens.disk == EllipticalSersic))
        >>> aggregator.filter((lens.bulge == EllipticalCoreSersic) | (lens.disk == EllipticalSersic))
        """
        return ListAggregator(
            self._fits_for_query(
                predicate.fit_query
            )
        )

    def _fits_for_query(
            self,
            query: str
    ) -> List[m.Fit]:
        """
        Execute a raw SQL query and return a Fit object
        for each Fit id returned by the query

        Parameters
        ----------
        query
            A SQL query that selects ids from the fit table

        Returns
        -------
        A list of fit objects, one for each id returned by the
        query
        """
        fit_ids = {
            row[0]
            for row
            in self.session.execute(
                query
            )
        }
        return self.session.query(
            m.Fit
        ).filter(
            m.Fit.id.in_(
                fit_ids
            )
        ).all()

    def add_directory(
            self,
            directory: str,
            auto_commit=True
    ):
        """
        Recursively search a directory for autofit results
        and add them to this database.

        Any pickles found in the pickles file are implicitly added
        to the fit object.

        Warnings
        --------
        If a directory is added twice then that will result in
        duplicate entries in the database.

        Parameters
        ----------
        auto_commit
            If True the session is committed writing the new objects
            to the database
        directory
            A directory containing autofit results embedded in a
            file structure
        """
        for fit in scrape_directory(
                directory
        ):

            #    fit.dataset_name = fit["dataset"].name
            self.session.add(
                fit
            )
        if auto_commit:
            self.session.commit()

    @classmethod
    def from_database(
            cls,
            filename: str
    ) -> "Aggregator":
        """
        Create an instance from a sqlite database file.

        If no file exists then one is created with the schema of the database.

        Parameters
        ----------
        filename
            The name of the database file.

        Returns
        -------
        An aggregator connected to the database specified by the file.
        """
        engine = create_engine(
            f'sqlite:///{filename}'
        )
        session = sessionmaker(
            bind=engine
        )()
        m.Base.metadata.create_all(
            engine
        )
        return Aggregator(
            session,
            filename
        )
