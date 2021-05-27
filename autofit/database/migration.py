from abc import ABC, abstractmethod
from hashlib import md5


class Identifiable(ABC):
    @property
    @abstractmethod
    def id(self):
        pass

    def __eq__(self, other):
        if isinstance(
                other,
                Identifiable
        ):
            return self.id == other.id
        if isinstance(
                other,
                str
        ):
            return self.id == other
        return False


class Migrator:
    def __init__(self, *steps):
        self.steps = steps

    @property
    def revisions(self):
        for i in range(1, len(self.steps) + 1):
            yield Revision(
                self.steps[:i]
            )


class Revision(Identifiable):
    def __init__(self, steps):
        self.steps = steps

    @property
    def id(self):
        return md5(
            ":".join(
                step.id for step
                in self.steps
            ).encode("utf-8")
        ).hexdigest()

    def __sub__(self, other):
        return Revision(tuple(
            step for step in self.steps
            if step not in other.steps
        ))


class Step(Identifiable):
    def __init__(self, string):
        self.string = string

    @property
    def id(self):
        return md5(
            self.string.encode(
                "utf-8"
            )
        ).hexdigest()
