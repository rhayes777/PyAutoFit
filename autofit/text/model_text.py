from configparser import NoOptionError
import logging

from autoconf import conf

logger = logging.getLogger(__name__)


def parameter_labels_from_model(model) -> [str]:
    """A list of every parameter's label, used by *GetDist* for model estimation and visualization.

    The parameter labels are determined using the label.ini and label_format.ini config files."""

    parameter_labels = []
    prior_class_dict = model.prior_class_dict

    for prior_name, prior in model.prior_tuples_ordered_by_id:
        try:
            param_string = conf.instance.label.label(prior_name)
        except NoOptionError:
            logger.warning(
                f"No label provided for {prior_name}. Using prior name instead."
            )
            param_string = prior_name

        cls = prior_class_dict[prior]
        cls_string = conf.instance.label.subscript(cls)

        parameter_label = "{}_{{\\mathrm{{{}}}}}".format(param_string, cls_string)
        parameter_labels.append(parameter_label)

    return parameter_labels