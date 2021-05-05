import shutil
from typing import Optional

from sqlalchemy.orm.exc import NoResultFound

from .abstract import AbstractPaths
from ...database.model import Fit


class DatabasePaths(AbstractPaths):
    def __init__(
            self,
            session,
            name=None,
            path_prefix=None,
            is_identifier_in_paths=True,
            parent=None,
            save_all_samples=False,
            unique_tag: Optional["str"] = None
    ):
        super().__init__(
            name=name,
            path_prefix=path_prefix,
            is_identifier_in_paths=is_identifier_in_paths,
            parent=parent
        )
        self.session = session
        self._fit = None
        self.save_all_samples = save_all_samples
        self.unique_tag = unique_tag

    parent: "DatabasePaths"

    def create_child(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            is_identifier_in_paths: Optional[bool] = None
    ) -> "DatabasePaths":
        """
        Create a paths object which is the child of some parent
        paths object. This is done during a GridSearch so that
        results can be stored in the correct directory. It also
        allows database fit objects to be related correctly.

        If no instance is set the prior median model is used
        to ensure that the parent object is queryable.

        Parameters
        ----------
        name
        path_prefix
        is_identifier_in_paths
            If False then this path's identifier will not be
            added to its output path.

        Returns
        -------
        A new paths object
        """
        self.fit.is_grid_search = True
        if self.fit.instance is None:
            self.fit.instance = self.model.instance_from_prior_medians()
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
        if self._fit is None:
            try:
                self._fit = self.session.query(
                    Fit
                ).filter(
                    Fit.id == self.identifier
                ).one()
            except NoResultFound:
                self._fit = Fit(
                    id=self.identifier,
                    is_complete=False,
                    unique_tag=self.unique_tag
                )
                self.session.add(
                    self._fit
                )

        if self.parent is not None:
            self._fit.parent = self.parent.fit
        return self._fit

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
        if not self.save_all_samples:
            samples = samples.minimise()

        self.fit.samples = samples

    def _load_samples(self):
        samples = self.fit.samples
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
