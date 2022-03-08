import shutil
from typing import Optional

from autofit.database.sqlalchemy_ import sa
from .abstract import AbstractPaths
from ...database.model import Fit


class DatabasePaths(AbstractPaths):
    def __init__(
            self,
            session,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
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

    @AbstractPaths.parent.setter
    def parent(
            self,
            parent: "DatabasePaths"
    ):
        """
        The search performed before this search. For example, a search
        that is then compared to searches during a grid search.

        For database paths the parent must also be database paths.
        """
        if not (
                parent is None or
                isinstance(parent, DatabasePaths)
        ):
            raise TypeError(
                "The parent of search that uses the database must also use the database"
            )
        self._parent = parent

    def save_named_instance(
            self,
            name: str,
            instance
    ):
        """
        Save an instance, such as that at a given sigma
        """
        self.fit.named_instances[
            name
        ] = instance

    @property
    def is_grid_search(self) -> bool:
        return self.fit.is_grid_search

    def create_child(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            is_identifier_in_paths: Optional[bool] = None,
            identifier: Optional[str] = None
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
        identifier
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
        child = type(self)(
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
        child.model = self.model
        child.search = self.search
        child._identifier = identifier
        return child

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

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["session"]
        return d

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
            except sa.orm.exc.NoResultFound:
                self._fit = Fit(
                    id=self.identifier,
                    is_complete=False,
                    unique_tag=self.unique_tag,
                    path_prefix=self.path_prefix,
                    name=self.name
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
        self.fit.max_log_likelihood = samples.max_log_likelihood_sample.log_likelihood
        super().save_summary(
            samples,
            log_likelihood_function_time
        )

    def save_samples(self, samples):
        if not self.save_all_samples:
            samples = samples.minimise()

        self.fit.samples = samples

    def samples_to_csv(self, samples):
        """
        Save the final-result samples associated with the phase as a pickle
        """
        pass

    def _load_samples(self):
        samples = self.fit.samples
        samples.model = self.model
        return samples

    def load_samples(self):
        return self._load_samples().sample_list

    def load_samples_info(self):
        return self._load_samples().info_json

    def save_all(self, info, *_, **kwargs):
        self.save_identifier()
        self.fit.info = info
        self.fit.model = self.model

        self.save_object("search", self.search)

        self.session.commit()
