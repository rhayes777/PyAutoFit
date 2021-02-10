from typing import Optional, List

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from autofit.aggregator.aggregator import Aggregator as ClassicAggregator
from autofit.database import query as q
from . import model as m


class Aggregator:
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

    def __iter__(self):
        return iter(
            self.fits
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.filename}>"

    def __getattr__(self, name):
        return q.Q(name)

    def __getitem__(self, item):
        return self.fits[0]

    def value(self, name):
        return [
            getattr(fit, name)
            for fit
            in self
        ]

    def query(self, predicate: q.Q) -> List[m.Object]:
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
        >>> from autogalaxy.profiles.light_profiles import EllipticalSersic, EllipticalCoreSersic
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
        query = f"SELECT id FROM fit WHERE instance_id IN ({predicate.query})"

        return self._fits_for_query(
            query
        )

    def _fits_for_query(self, query: str):
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

    def __len__(self):
        return self.session.query(
            m.Fit
        ).count()

    def add_directory(self, directory: str):
        """
        Recursively search a directory for autofit results
        and add them to this database.

        Warnings
        --------
        If a directory is added twice then that will result in
        duplicate entries in the database.

        Parameters
        ----------
        directory
            A directory containing autofit results embedded in a
            file structure
        """
        aggregator = ClassicAggregator(
            directory
        )
        for item in aggregator:
            model = m.Object.from_object(
                item.model
            )
            samples = item.samples
            instance = m.Object.from_object(
                samples.max_log_likelihood_instance
            )
            fit = m.Fit(
                model=model,
                instance=instance,
                samples=m.Object.from_object(
                    samples
                )
            )
            self.session.add(
                fit
            )
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
