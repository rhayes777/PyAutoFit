import shutil

from sqlalchemy.orm.exc import NoResultFound

from .abstract import AbstractPaths
from ...database.model import Fit, Object


class DatabasePaths(AbstractPaths):
    def __init__(
            self,
            session,
            name=None,
            path_prefix=None,
            is_identifier_in_paths=True,
            parent=None
    ):
        super().__init__(
            name=name,
            path_prefix=path_prefix,
            is_identifier_in_paths=is_identifier_in_paths,
            parent=parent
        )
        self.session = session

    parent: "DatabasePaths"

    def create_child(
            self,
            name=None,
            path_prefix=None,
            is_identifier_in_paths=None
    ):
        self.fit.is_grid_search = True
        return type(self)(
            session=self.session,
            name=name or self.name,
            path_prefix=path_prefix or self.path_prefix,
            is_identifier_in_paths=(
                is_identifier_in_paths
                if is_identifier_in_paths is not None
                else self.is_identifier_in_paths
            ),
            parent=self
        )

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
        self.fit[name] = obj

    def load_object(self, name: str):
        return self.fit[name]

    def remove_object(self, name: str):
        del self.fit[name]

    def is_object(self, name: str) -> bool:
        return name in self.fit

    @property
    def fit(self) -> Fit:
        try:
            fit = self.session.query(
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

        if self.parent is not None:
            fit.parent = self.parent.fit
        return fit

    @property
    def is_complete(self) -> bool:
        return self.fit.is_complete

    def completed(self):
        self.fit.is_complete = True

    def save_summary(
            self,
            samples,
            log_likelihood_function_time
    ):
        self.fit.instance = samples.max_log_likelihood_instance
        super().save_summary(
            samples,
            log_likelihood_function_time
        )

    def save_samples(self, samples):
        self.fit.samples = Object.from_object(
            samples.minimise()
        )

    def _load_samples(self):
        samples = self.fit.samples()
        samples.model = self.model
        return samples

    def load_samples(self):
        return self._load_samples().samples

    def load_samples_info(self):
        return self._load_samples().info_json

    def save_all(self, info, *_, **kwargs):
        self.fit.info = info
        self.fit.model = self.model

        if self.search is not None:
            self.search.paths = None
        self.save_object("search", self.search)
        if self.search is not None:
            self.search.paths = self

        self.session.commit()
