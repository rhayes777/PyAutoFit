import inspect
import itertools
from collections import Iterable
from hashlib import md5

from autofit.util import get_class

# floats are rounded to this increment so floating point errors
# have no impact on identifier value
RESOLUTION = 1e-8


class Identifier:
    def __init__(self, obj):
        """
        Wraps an object and recursively generates an identifier
        """
        self.hash_list = list()
        self._add_value_to_hash_list(
            obj
        )

    @property
    def description(self):
        return "\n".join(self.hash_list)

    def _add_value_to_hash_list(self, value):
        """
        Add some object and recursively add its children to the hash_list.

        The md5 hash of this object is taken to create an identifier.

        If an object specifies __identifier_fields__ then only attributes
        with a name in this list are included.

        Parameters
        ----------
        value
            An object
        """
        if hasattr(value, "__dict__"):
            d = value.__dict__
            if hasattr(
                    value,
                    "__identifier_fields__"
            ):
                fields = value.__identifier_fields__
                missing_fields = [
                    field for field in fields
                    if field not in d
                ]
                if len(missing_fields) > 0:
                    string = '\n'.join(
                        missing_fields
                    )
                    raise AssertionError(
                        f"The following __identifier_fields__ do not exist for {type(value)}:\n{string}"
                    )
                d = {
                    k: v
                    for k, v
                    in d.items()
                    if k in fields
                }
            elif hasattr(
                    value,
                    "__class__"
            ) and not inspect.isclass(
                value
            ) and not isinstance(
                value,
                ModelObject
            ):
                args = inspect.getfullargspec(
                    value.__class__
                ).args
                d = {
                    k: v
                    for k, v
                    in d.items()
                    if k in args
                }
            self.add_value_to_hash_list(
                d
            )
        elif isinstance(
                value, dict
        ):
            for key, value in value.items():
                if not (key.startswith("_") or key in ("id", "paths")):
                    self.hash_list.append(key)
                    self.add_value_to_hash_list(
                        value
                    )
        elif isinstance(
                value, float
        ):
            try:
                value = RESOLUTION * round(
                    value / RESOLUTION
                )
            except OverflowError:
                pass

            self.hash_list.append(
                str(value)
            )
        elif isinstance(
                value,
                (str, int, bool)
        ):
            self.hash_list.append(
                str(value)
            )
        elif isinstance(value, Iterable):
            for value in value:
                self.add_value_to_hash_list(
                    value
                )

    def add_value_to_hash_list(self, value):
        if isinstance(
                value,
                property
        ):
            return
        self._add_value_to_hash_list(
            value
        )

    def __str__(self):
        return md5(".".join(
            self.hash_list
        ).encode("utf-8")).hexdigest()

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    def __eq__(self, other):
        return str(self) == str(other)


class ModelObject:
    _ids = itertools.count()

    def __init__(self):
        self.id = next(self._ids)

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
        from autofit.mapper.prior.prior import Prior
        from autofit.mapper.prior.prior import TuplePrior

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

    @property
    def dict(self) -> dict:
        """
        A dictionary representation of this object
        """
        from autofit.mapper.prior_model.abstract import AbstractPriorModel
        from autofit.mapper.prior_model.collection import CollectionPriorModel
        from autofit.mapper.prior_model.prior_model import PriorModel
        from autofit.mapper.prior.prior import TuplePrior

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
                value = value.dict
            except AttributeError:
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
