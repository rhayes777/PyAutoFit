import copy


class AbstractModel(object):
    def __add__(self, other):
        instance = self.__class__()

        def add_items(item_dict):
            for key, value in item_dict.items():
                if isinstance(value, list) and hasattr(instance, key):
                    setattr(instance, key, getattr(instance, key) + value)
                else:
                    setattr(instance, key, value)

        add_items(self.__dict__)
        add_items(other.__dict__)
        return instance

    def copy(self):
        return copy.deepcopy(self)

    def object_for_path(self, path):
        instance = self
        for name in path:
            instance = getattr(instance, name)
        return instance


class ModelInstance(AbstractModel):
    """
    An object to hold model instances produced by providing arguments to a model mapper.

    @DynamicAttrs
    """

    def instances_of(self, cls):
        return [instance for source in
                [list(self.__dict__.values())] + [ls for ls in self.__dict__.values() if
                                                  isinstance(ls, list)] for
                instance in
                source if isinstance(instance, cls)]

    def name_instance_tuples_for_class(self, cls):
        return [
            (
                "_".join(
                    item[0]
                ),
                item[1]
            ) for item in self.path_instance_tuples_for_class(cls)]

    def path_instance_tuples_for_class(self, cls):
        flat = [((item[0],), item[1]) for item in self.__dict__.items() if
                isinstance(item[1], cls)]
        if not cls == ModelInstance:
            sub_instances = self.path_instance_tuples_for_class(ModelInstance)
            sub = [((*instance[0], *item[0]), item[1]) for instance in
                   sub_instances for item in
                   instance[1].path_instance_tuples_for_class(cls)]
            return flat + sub
        return flat

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __getitem__(self, item):
        return self.items[item]

    @property
    def items(self):
        return [value for key, value in self.__dict__.items() if
                key not in ("id", "component_number", "item_number")]

    def __len__(self):
        return len(self.items)
