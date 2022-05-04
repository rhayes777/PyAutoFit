import numpy as np

from autoconf import conf
from autofit import exc
from autofit.messages.normal import NormalMessage, UniformNormalMessage, LogNormalMessage
from autofit.messages.transform import log_10_transform
from autofit.messages.transform_wrapper import TransformedWrapperInstance
from .abstract import Prior
from .abstract import epsilon


class Limits:
    @staticmethod
    def for_class_and_attributes_name(cls, attribute_name):
        limit_dict = conf.instance.prior_config.for_class_and_suffix_path(
            cls, [attribute_name, "gaussian_limits"]
        )
        return limit_dict["lower"], limit_dict["upper"]


class WrappedInstance(
    TransformedWrapperInstance
):
    __identifier_fields__ = ("lower_limit", "upper_limit")

    __database_args__ = (
        "lower_limit",
        "upper_limit",
        "log_norm",
        "id_",
    )

    def __init__(
            self,
            transformed_wrapper,
            *args,
            lower_limit,
            upper_limit,
            **kwargs
    ):
        super().__init__(
            transformed_wrapper,
            *args,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            **kwargs
        )
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        if self.lower_limit >= self.upper_limit:
            raise exc.PriorException(
                "The upper limit of a prior must be greater than its lower limit"
            )

    def _new_for_base_message(
            self,
            message
    ):
        return type(self)(
            self.transformed_wrapper,
            *message.parameters,
            lower_limit=self.lower_limit,
            upper_limit=self.upper_limit,
            id_=self.instance().id,
        )
