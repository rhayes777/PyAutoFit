import shutil

from sqlalchemy.orm.exc import NoResultFound

from .abstract import AbstractPaths
from ...database.model import Fit, Object


class DatabasePaths(AbstractPaths):
    def __init__(
            self,
            session,
            name=None,
            path_prefix=None
    ):
        super().__init__(
            name=name,
            path_prefix=path_prefix,
        )
        self.session = session

    def zip_remove(self):
        """
        Remove files from both the symlinked folder and the output directory
        """
        self.session.commit()

        if self.remove_files:
            shutil.rmtree(
                self.path,
                ignore_errors=True
            )
            shutil.rmtree(
                self.output_path,
                ignore_errors=True
            )

    def save_object(self, name: str, obj: object):
        self._fit[name] = obj

    def load_object(self, name: str):
        return self._fit[name]

    def remove_object(self, name: str):
        del self._fit[name]

    def is_object(self, name: str) -> bool:
        return name in self._fit

    @property
    def _fit(self) -> Fit:
        try:
            return self.session.query(
                Fit
            ).filter(
                Fit.id == self.identifier
            ).one()
        except NoResultFound:
            fit = Fit(
                id=self.identifier,
                is_complete=False
            )
            self.session.add(
                fit
            )
            return fit

    @property
    def is_complete(self) -> bool:
        return self._fit.is_complete

    def completed(self):
        self._fit.is_complete = True

    def save_summary(
            self,
            samples,
            log_likelihood_function_time
    ):
        self._fit.instance = samples.max_log_likelihood_instance
        super().save_summary(
            samples,
            log_likelihood_function_time
        )

    def save_samples(self, samples):
        self._fit.samples = Object.from_object(
            samples.minimise()
        )

    def _load_samples(self):
        samples = self._fit.samples()
        samples.model = self.model
        return samples

    def load_samples(self):
        return self._load_samples().samples

    def load_samples_info(self):
        return self._load_samples().info_json

    def save_all(self, info, *_, **kwargs):
        self._fit.info = info
        self._fit.model = self.model

        if self.search is not None:
            self.search.paths = None
        self.save_object("search", self.search)
        if self.search is not None:
            self.search.paths = self

        self.session.commit()
