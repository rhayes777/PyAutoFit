from pathlib import Path
from typing import Union, List

from sqlalchemy.orm import Session

from autofit.database.model.fit import Fit
from autofit.text.formatter import write_table


class Info:
    def __init__(self, session: Session):
        """
        Write info describing the contents of a database file
        in terms of the fits it contains.

        Parameters
        ----------
        session
            A database session
        """
        self.session = session

    @property
    def fits(self) -> List[Fit]:
        """
        A list of objects representing all the fits in the database.
        """
        return self.session.query(Fit).all()

    @property
    def headers(self) -> List[str]:
        """
        A list of strings describing the columns of the table.
        """
        return [
            "unique_id",
            "name",
            "unique_tag",
            "total_free_parameters",
            "is_complete",
        ]

    @property
    def rows(self) -> List[List[str]]:
        """
        A list of lists of strings describing the rows of the table.
        """
        return [
            [
                fit.id,
                fit.name,
                fit.unique_tag,
                fit.total_parameters,
                fit.is_complete,
            ]
            for fit in self.fits
        ]

    def write(self, filename: Union[str, Path]):
        """
        Write the info to a file.
        """
        write_table(self.headers, self.rows, filename)
