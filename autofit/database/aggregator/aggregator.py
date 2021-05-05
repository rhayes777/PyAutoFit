from typing import Optional, List, Union

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from autofit.database import query as q
from .scrape import scrape_directory
from .. import model as m
from ..query.query import AbstractQuery


class NullPredicate(AbstractQuery):
    @property
    def fit_query(self) -> str:
        return "SELECT id FROM fit"

    def __and__(self, other):
        return other


class Query:
    @staticmethod
    def for_name(name):
        if name in m.fit_attributes:
            if m.fit_attributes[
                name
            ].type.python_type == bool:
                return q.BA(name)
            return q.A(name)
        return q.Q(name)

    def __getattr__(self, name):
        return self.for_name(name)


class Aggregator:
    def __init__(
            self,
            session: Session,
            filename: Optional[str] = None,
            predicate: AbstractQuery = NullPredicate(),
            offset=0,
            limit=None
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
        self._predicate = predicate
        self._offset = offset
        self._limit = limit

    def __iter__(self):
        return iter(
            self.fits
        )

    @property
    def info(self):
        """
        Query info associated with the fit in the info dictionary
        """
        return q.AnonymousInfo()

    def values(self, name: str) -> list:
        """
        Retrieve the value associated with each fit with the given
        parameter name

        Parameters
        ----------
        name
            The name of some pickle, such as 'samples'

        Returns
        -------
        A list of objects, one for each fit
        """
        return [
            fit[name]
            for fit
            in self
        ]

    def __len__(self):
        return len(self.fits)

    def __eq__(self, other):
        if isinstance(other, list):
            return self.fits == other
        return super().__eq__(other)

    @property
    def fits(self) -> List[m.Fit]:
        """
        Lazily query the database for a list of Fit objects that
        match the aggregator's predicate.
        """
        if self._fits is None:
            self._fits = self._fits_for_query(
                self._predicate.fit_query
            )
        return self._fits

    def map(self, func):
        for fit in self.fits:
            yield func(fit)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.filename} {len(self)}>"

    def __getattr__(self, name: str) -> Union[AbstractQuery, q.A]:
        """
        Facilitates query construction. If the Fit class has an
        attribute with the given name then a predicate is generated
        based on that attribute. Otherwise the query is assumed to
        apply to the best fit instance.

        Parameters
        ----------
        name
            The name of an attribute of the Fit class or the model

        Returns
        -------
        A query
        """
        return Query.for_name(name)

    def __call__(self, predicate) -> "Aggregator":
        """
        Concise query syntax
        """
        return self.query(predicate)

    def query(self, predicate: AbstractQuery) -> "Aggregator":
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
        >>> aggregator.filter((lens.bulge == EllSersicCore) & (lens.disk == EllSersic))
        >>> aggregator.filter((lens.bulge == EllSersicCore) | (lens.disk == EllSersic))
        """
        return self._new_with(
            predicate=self._predicate & predicate
        )

    def _new_with(
            self,
            **kwargs
    ):
        kwargs = {
            "session": self.session,
            "filename": self.filename,
            "predicate": self._predicate,
            **kwargs
        }
        return Aggregator(
            **kwargs
        )

    def children(self) -> "Aggregator":
        """
        An aggregator comprising the children of the fits encapsulated
        by this aggregator. This is used to query children in a grid search.
        """
        return Aggregator(
            session=self.session,
            filename=self.filename,
            predicate=q.ChildQuery(
                self._predicate
            )
        )

    def __getitem__(self, item):
        offset = self._offset
        limit = self._limit
        if isinstance(
                item, int
        ):
            return self.fits[item]
        elif isinstance(
                item, slice
        ):
            if item.start is not None:
                if item.start >= 0:
                    offset += item.start
                else:
                    offset = len(self) + item.start
            if item.stop is not None:
                if item.stop >= 0:
                    limit = len(self) - item.stop - offset
                else:
                    limit = len(self) + item.stop
        return self._new_with(
            offset=offset,
            limit=limit
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
        ).offset(
            self._offset
        ).limit(
            self._limit
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
            self.session.add(
                fit
            )
        if auto_commit:
            self.session.commit()

    @classmethod
    def from_database(
            cls,
            filename: str,
            completed_only: bool = False
    ) -> "Aggregator":
        """
        Create an instance from a sqlite database file.

        If no file exists then one is created with the schema of the database.

        Parameters
        ----------
        completed_only
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
        aggregator = Aggregator(
            session,
            filename
        )
        if completed_only:
            return aggregator(
                aggregator.is_complete
            )
        return aggregator
