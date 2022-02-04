import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Union, cast

from ..sqlalchemy_ import sa

from autofit.database import query as q
from .scrape import Scraper
from autofit.database import model as m
from ..query.query import AbstractQuery, Attribute
from ..query.query.attribute import BestFitQuery

logger = logging.getLogger(
    __name__
)


class NullPredicate(AbstractQuery):
    @property
    def fit_query(self) -> str:
        return "SELECT id FROM fit"

    def __and__(self, other):
        return other


class Query:
    """
    API for creating a query on the best fit instance
    """

    @staticmethod
    def for_name(name: str) -> q.Q:
        """
        Create a query for fits based on the name of a
        top level instance attribute

        Parameters
        ----------
        name
            The name of the attribute. e.g. galaxies

        Returns
        -------
        A query generating object
        """
        return q.Q(name)

    def __getattr__(self, name):
        return self.for_name(name)


class FitQuery(Query):
    """
    API for creating a query on the attributes of a fit,
    such as:
        name
        unique_tag
        path_prefix
        is_complete
        is_grid_search
    """

    @staticmethod
    def for_name(name: str) -> Union[
        AbstractQuery,
        Attribute
    ]:
        """
        Create a query based on some attribute of the Fit.

        Parameters
        ----------
        name
            The name of an attribute of the Fit class

        Returns
        -------
        A query based on an attribute

        Examples
        --------
        aggregator.fit.name == 'example name'
        """
        if name not in m.fit_attributes:
            raise AttributeError(
                f"Fit has no attribute {name}"
            )
        if m.fit_attributes[
            name
        ].type.python_type == bool:
            return q.BA(name)
        return q.A(name)


class Reverse:
    def __init__(self, item):
        self.item = item

    @property
    def attribute(self):
        return self.item.attribute


class AbstractAggregator(ABC):
    @property
    @abstractmethod
    def fits(self) -> List[m.Fit]:
        pass

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
        values = list()
        for fit in self:
            value = fit[name]
            if value is not None:
                values.append(value)

        return values

    def __iter__(self):
        return iter(
            self.fits
        )

    def __len__(self):
        return len(self.fits)

    def __eq__(self, other):
        if isinstance(other, list):
            return self.fits == other
        return super().__eq__(other)


class Aggregator(AbstractAggregator):
    def __init__(
            self,
            session: sa.orm.Session,
            filename: Optional[str] = None,
            predicate: AbstractQuery = NullPredicate(),
            offset=0,
            limit=None,
            order_bys=None
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
        self._order_bys = order_bys or list()

    def order_by(
            self,
            item: Attribute,
            reverse=False
    ) -> "Aggregator":
        """
        Order the results by a given attribute of the search. Can be applied
        multiple times with the first application taking precedence.

        Parameters
        ----------
        item
            An attribute of the search
        reverse
            If True reverse the results

        Returns
        -------
        An aggregator with ordering applied

        Examples
        --------
        aggregator = aggregator.order_by(
            aggregator.search.unique_tag
        )
        """
        if reverse:
            item = Reverse(item)
        return self._new_with(
            order_bys=self._order_bys + [item]
        )

    @property
    def search(self) -> FitQuery:
        """
        An object facilitating queries on fit attributes such as:
            name
            unique_tag
            path_prefix
            is_complete
            is_grid_search
        """
        return FitQuery()

    @property
    def info(self):
        """
        Query info associated with the fit in the info dictionary
        """
        return q.AnonymousInfo()

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

    @property
    def model(self) -> Query:
        """
        Facilitates query construction. If the Fit class has an
        attribute with the given name then a predicate is generated
        based on that attribute. Otherwise the query is assumed to
        apply to the best fit instance.

        Returns
        -------
        A query
        """
        return Query()

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
            type_=None,
            **kwargs,
    ) -> "Aggregator":
        """
        Create a new instance with the same attribute values except
        for those overridden by kwargs

        Parameters
        ----------
        type_
            The type of the new instance (defaults to Aggregator)
        kwargs
            Names and values of attributes to override

        Returns
        -------
        A new Aggregator with the same attributes except where they
        have been overridden
        """
        kwargs = {
            "session": self.session,
            "filename": self.filename,
            "predicate": self._predicate,
            "order_bys": self._order_bys,
            **kwargs
        }
        type_ = type_ or type(self)
        return type_(
            **kwargs
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
        logger.debug(
            f"Executing query: {query}"
        )
        fit_ids = {
            row[0]
            for row
            in self.session.execute(
                query
            )
        }

        logger.info(
            f"{len(fit_ids)} fit(s) found matching query"
        )
        query = self.session.query(
            m.Fit
        ).filter(
            m.Fit.id.in_(
                fit_ids
            )
        )
        for order_by in self._order_bys:
            attribute = getattr(
                m.Fit,
                order_by.attribute
            )

            if isinstance(
                    order_by,
                    Reverse
            ):
                attribute = sa.desc(attribute)
            query = query.order_by(
                attribute
            )

        return query.offset(
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
        scraper = Scraper(
            directory,
            self.session
        )
        scraper.scrape()

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
        from autofit.database import open_database
        session = open_database(
            str(filename)
        )
        aggregator = Aggregator(
            session,
            filename
        )
        if completed_only:
            return aggregator(
                aggregator.search.is_complete
            )
        return aggregator

    def grid_searches(self) -> "GridSearchAggregator":
        """
        Filter to only grid searches and return an aggregator
        with grid search specific functionality.

        Grid searches are initially implicitly ordered by their id
        """
        return cast(
            GridSearchAggregator,
            self._new_with(
                type_=GridSearchAggregator,
                predicate=self._predicate & self.search.is_grid_search,
                order_bys=[Attribute("id")]
            ),
        )


class GridSearchAggregator(Aggregator):
    def best_fits(self) -> "GridSearchAggregator":
        """
        The best fit from each of the grid searches

        Best fits are initially implicitly ordered by their parent id
        """
        return self._new_with(
            predicate=BestFitQuery(
                self._predicate
            ),
            order_bys=[Attribute("parent_id")]
        )

    def children(self) -> "GridSearchAggregator":
        """
        An aggregator comprising the children of the fits encapsulated
        by this aggregator. This is used to query children in a grid search.

        Children are initially implicitly ordered by their parent id
        """
        return self._new_with(
            predicate=q.ChildQuery(
                self._predicate
            ),
            order_bys=[Attribute("parent_id")]
        )

    def cell_number(
            self,
            number: int
    ) -> "CellAggregator":
        """
        Create an aggregator for accessing all values for child fits
        with a given index, ordered by parameter values.

        Parameters
        ----------
        number
            The number of the fit in the grid search

        Returns
        -------
        An aggregator comprising fits for a given cell for each grid search
        """
        return CellAggregator(
            number,
            self
        )


class CellAggregator(AbstractAggregator):
    def __init__(
            self,
            number: int,
            aggregator: GridSearchAggregator
    ):
        """
        Aggregator for accessing data for a specific fit number in each
        grid search.

        Parameters
        ----------
        number
            The number of the fit
        aggregator
            An aggregator comprising 0 or more grid searches
        """
        self.number = number
        self.aggregator = aggregator
        self._fits = None

    @property
    def fits(self) -> List[m.Fit]:
        """
        Retrieve one fit for each grid search matching the number of
        the cell.
        """
        if self._fits is None:
            self._fits = list()
            for fit in self.aggregator:
                self._fits.append(
                    sorted(
                        fit.children,
                        key=lambda f: f.model.order_no if f.model is not None else 0
                    )[self.number]
                )
        return self._fits
