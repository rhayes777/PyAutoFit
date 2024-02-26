from typing import List, Iterable

import autofit as af


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
    def find_representatives(cls, items: Iterable[tuple], minimum: int = 4) -> list:
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

        for key, obj in sorted(items):
            blueprint = Blueprint(obj).blueprint
            if blueprint == last_blue_print:
                current_items.append((key, obj))
            else:
                add()
                current_items = [(key, obj)]
                last_blue_print = blueprint

        add()

        return representatives


class Blueprint:
    def __init__(self, top_level):
        from autofit.text.formatter import FormatNode

        if isinstance(top_level, FormatNode):
            top_level = top_level.value
        self.top_level = top_level

    def get_blueprint(self, obj):
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
        if obj is None:
            return None

        if isinstance(obj, (float, int, tuple, str)):
            return obj
        if isinstance(obj, af.Prior):
            blueprint = (
                type(obj),
                obj.parameter_string,
            )
            try:
                blueprint += tuple(self.top_level.all_paths_to_child(obj))
            except AttributeError:
                pass
            return blueprint
        if isinstance(obj, af.AbstractModel):
            blueprint = tuple(
                (path, self.get_blueprint(value))
                for path, value in obj.path_instance_tuples_for_class(
                    (float, int, tuple, af.Prior), ignore_children=True
                )
                if path != ("id",)
            )
            path_priors = obj.path_instance_tuples_for_class(
                af.Prior, ignore_children=True
            )
            min_id = min(pp[1].id for pp in path_priors)
            blueprint += tuple(
                (path, prior.id - min_id, self.get_blueprint(prior))
                for path, prior in obj.path_instance_tuples_for_class(
                    af.Prior, ignore_children=True
                )
            )
            if isinstance(obj, af.Model):
                return blueprint + (obj.cls,)
            return blueprint
        raise ValueError(f"Cannot get blueprint for {obj} of type {type(obj)}")

    @property
    def blueprint(self):
        """
        A dictionary of blueprints for each object in the model.
        """
        return self.get_blueprint(self.top_level)
