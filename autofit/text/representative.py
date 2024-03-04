from collections import Counter
from typing import List, Iterable, Set

import autofit as af
from autofit.mapper.prior.abstract import Prior


def is_prior(obj):
    from autofit.text.formatter import FormatNode
    from autofit.mapper.prior.arithmetic.compound import CompoundPrior

    if isinstance(obj, FormatNode):
        try:
            return is_prior(obj.value)
        except AttributeError:
            return False
    return isinstance(obj, (Prior, CompoundPrior))


class Representative:
    def __init__(self, items: List[tuple]):
        """
        Collects together items that are the same except for ids. This is to make
        the output of the info method more concise.

        Parameters
        ----------
        items
            A list of tuples of the form (key, object)
        """
        self.items = items

    @property
    def keys(self):
        return [key for key, _ in self.items]

    @property
    def children(self):
        return [obj for _, obj in self.items]

    @property
    def key(self) -> str:
        """
        A string representation of the range of keys in this representative.
        """
        keys = sorted(self.keys)
        return f"{keys[0]} - {keys[-1]}"

    @property
    def representative(self):
        """
        The first object in the group represents the group.
        """
        try:
            return sorted(self.children)[0]
        except TypeError:
            return self.children[0]

    def __getattr__(self, item):
        return getattr(self.representative, item)

    def __len__(self):
        return len(self.representative)

    @classmethod
    def find_representatives(cls, items: Iterable[tuple], minimum: int = 2) -> list:
        """
        Find representatives in a list of items. This includes items from
        the original list where there are not enough repetitions to form
        a representative.

        Parameters
        ----------
        items
            A list of tuples of the form (key, object)
        minimum
            The minimum number of items that must be the same for a representative
            to be formed.

        Returns
        -------
        A list of representatives and items that are not part of a representative.
        """
        representatives = []
        last_blue_print = None
        current_items = []

        def add():
            if len(current_items) >= minimum:
                representative = Representative(current_items)
                representatives.append((representative.key, representative))
            else:
                representatives.extend(current_items)

        shared_priors = cls.shared_descendents(obj for _, obj in items)

        for key, obj in sorted(items):
            try:
                if is_prior(obj) or any(
                    prior in shared_priors
                    for _, prior in obj.path_instance_tuples_for_class(af.Prior)
                ):
                    add()
                    current_items = [(key, obj)]
                    last_blue_print = None
                    continue
            except AttributeError:
                pass

            blueprint = cls.get_blueprint(obj)
            if blueprint is not None and blueprint == last_blue_print:
                current_items.append((key, obj))
            else:
                add()
                current_items = [(key, obj)]
                last_blue_print = blueprint

        add()

        return representatives

    @classmethod
    def get_blueprint(cls, obj):
        """
        Get a blueprint for an object. This is a tuple of tuples of the form
        (path, value) where path is a tuple of strings and value is a float, int,
        tuple, or af.Prior.

        Blueprints are unique per unique object in the model but are not sensitive
        to ids.

        Parameters
        ----------
        obj
            The object to get a blueprint for.

        Returns
        -------
        A blueprint for the object.
        """
        from autofit.text.formatter import FormatNode

        if obj is None:
            return ()

        if isinstance(obj, FormatNode):
            blueprint = cls.get_blueprint(obj.value)
            for key, value in obj.items():
                blueprint += (key,)
                blueprint += cls.get_blueprint(value)
            return blueprint
        if isinstance(obj, (float, int, tuple, str)):
            return (obj,)
        if isinstance(obj, af.Prior):
            return type(obj), obj.parameter_string
        if isinstance(obj, af.AbstractModel):
            blueprint = tuple(
                (path, cls.get_blueprint(value))
                for path, value in obj.path_instance_tuples_for_class(
                    (float, int, tuple, af.Prior), ignore_children=True
                )
                if path != ("id",)
            )
            path_priors = obj.path_instance_tuples_for_class(
                af.Prior, ignore_children=True
            )
            try:
                min_id = min(pp[1].id for pp in path_priors)
            except ValueError:
                min_id = 0
            blueprint += tuple(
                (path, prior.id - min_id, cls.get_blueprint(prior))
                for path, prior in obj.path_instance_tuples_for_class(
                    af.Prior, ignore_children=True
                )
            )
            if isinstance(obj, af.Model):
                return blueprint + (obj.cls,)
            return blueprint

        return (type(obj),) + tuple(
            (key, value)
            for key, value in obj.__dict__.items()
            if key != "id" and isinstance(value, (float, int))
        )

    @classmethod
    def shared_descendents(cls, objects) -> Set[Prior]:
        """
        Find all priors which are shared by more than one object in a list of items.

        Parameters
        ----------
        objects
            A list of objects.

        Returns
        -------
        A set of priors shared by more than one object.
        """
        counts = Counter()
        for obj in objects:
            try:
                for _, prior in obj.path_instance_tuples_for_class(
                    af.Prior, ignore_children=True
                ):
                    counts[prior] += 1
            except AttributeError:
                pass

        return {prior for prior, count in counts.items() if count > 1}
