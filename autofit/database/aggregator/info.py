from autofit.database.model.fit import Fit


class Info:
    def __init__(self, session):
        self.session = session

    @property
    def fits(self):
        return self.session.query(Fit).all()

    @property
    def headers(self):
        return [
            "unique_id",
            "name",
            "unique_tag",
            "total_free_parameters",
            "is_complete",
        ]

    @property
    def rows(self):
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
