from typing import Optional
from sqlalchemy.orm import Session

from autoconf import conf
from autofit.non_linear.abstract_search import NonLinearSearch


class AbstractMLE(NonLinearSearch):

    def __init__(
            self,
            name=None,
            path_prefix=None,
            unique_tag : Optional[str] = None,
            prior_passer=None,
            initializer=None,
            iterations_per_update : int = None,
            session : Optional[Session] = None,
            **kwargs
    ):

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            prior_passer=prior_passer,
            initializer=initializer,
            iterations_per_update=iterations_per_update,
            session=session,
            **kwargs
        )

    @property
    def config_type(self):
        return conf.instance["non_linear"]["mle"]
