from .collection import Collection


class Representative(Collection):
    def __init__(self, obj):
        keys = sorted(obj.keys())
        super().__init__({f"{keys[0]} - {keys[-1]}": obj[0]})
