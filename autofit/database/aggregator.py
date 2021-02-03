from typing import Optional, List

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from autofit.aggregator.aggregator import Aggregator as ClassicAggregator
from autofit.database import query_model as q
from .model import Object, Base


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

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.filename}>"

    def __getattr__(self, name):
        return q.Q(name)

    def query(self, predicate: q.Q) -> List[Object]:
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
        objects_ids = {
            row[0]
            for row
            in self.session.execute(
                predicate.query
            )
        }
        return self.session.query(
            Object
        ).filter(
            Object.id.in_(
                objects_ids
            )
        ).all()

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
            obj = Object.from_object(
                item.model
            )
            self.session.add(
                obj
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
        Base.metadata.create_all(
            engine
        )
        return Aggregator(
            session,
            filename
        )
