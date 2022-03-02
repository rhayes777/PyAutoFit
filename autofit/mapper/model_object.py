import itertools

from autoconf.class_path import get_class
from .identifier import Identifier


class ModelObject:
    _ids = itertools.count()

    def __init__(
            self,
            id_=None,
            label=None,
    ):
        """
        A generic object in AutoFit

        Parameters
        ----------
        id_
            A unique integer identifier. This is used to hash and order priors.
        label
            A label which can optionally be set for visualising this object in a
            graph.
        """
        self.id = next(self._ids) if id_ is None else id_
        self._label = label

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def component_number(self):
        return self.id

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        try:
            return self.id == other.id
        except AttributeError:
            return False

    @property
    def identifier(self):
        return str(Identifier(self))

    @staticmethod
    def from_dict(d):
        """
        Recursively parse a dictionary returning the model, collection or
        instance that is represents.

        Parameters
        ----------
        d
            A dictionary representation of some object

        Returns
        -------
        An instance
        """
        from autofit.mapper.prior_model.abstract import AbstractPriorModel
        from autofit.mapper.prior_model.collection import CollectionPriorModel
        from autofit.mapper.prior_model.prior_model import PriorModel
        from autofit.mapper.prior.abstract import Prior
        from autofit.mapper.prior.tuple_prior import TuplePrior

        if not isinstance(
                d, dict
        ):
            return d

        type_ = d["type"]

        if type_ == "model":
            instance = PriorModel(
                get_class(
                    d.pop("class_path")
                )
            )
        elif type_ == "collection":
            instance = CollectionPriorModel()
        elif type_ == "instance":
            cls = get_class(
                d.pop("class_path")
            )
            instance = object.__new__(cls)
        elif type_ == "tuple_prior":
            instance = TuplePrior()
        else:
            return Prior.from_dict(d)

        d.pop("type")

        for key, value in d.items():
            setattr(
                instance,
                key,
                AbstractPriorModel.from_dict(value)
            )
        return instance

    def dict(self) -> dict:
        """
        A dictionary representation of this object
        """
        from autofit.mapper.prior_model.abstract import AbstractPriorModel
        from autofit.mapper.prior_model.collection import CollectionPriorModel
        from autofit.mapper.prior_model.prior_model import PriorModel
        from autofit.mapper.prior.tuple_prior import TuplePrior

        if isinstance(
                self,
                CollectionPriorModel
        ):
            type_ = "collection"
        elif isinstance(
                self,
                AbstractPriorModel
        ) and self.prior_count == 0:
            type_ = "instance"
        elif isinstance(
                self,
                PriorModel
        ):
            type_ = "model"
        elif isinstance(
                self,
                TuplePrior
        ):
            type_ = "tuple_prior"
        else:
            raise AssertionError(
                f"{self.__class__.__name__} cannot be serialised to dict"
            )

        dict_ = {
            "type": type_
        }

        for key, value in self._dict.items():
            try:
                if not isinstance(
                        value, ModelObject
                ):
                    value = AbstractPriorModel.from_instance(
                        value
                    )
                value = value.dict()
            except AttributeError:
                pass
            except TypeError:
                pass
            dict_[key] = value
        return dict_

    @property
    def _dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("component_number", "item_number", "id", "cls")
               and not key.startswith("_")
        }
