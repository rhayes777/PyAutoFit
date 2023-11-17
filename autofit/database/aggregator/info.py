from autofit.database.model.fit import Fit


class Info:
    def __init__(self, session):
        self.session = session

    @property
    def fits(self):
        return self.session.query(Fit).all()
