from abc import ABC, abstractmethod
from hashlib import md5

import pytest


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


@pytest.fixture(
    name="step_1"
)
def make_step_1():
    return Step(
        "ALTER TABLE fit ADD name VARCHAR"
    )


@pytest.fixture(
    name="step_2"
)
def make_step_2():
    return Step(
        "ALTER TABLE fit ADD path_prefix VARCHAR"
    )


def test_step_id(
        step_1,
        step_2
):
    assert isinstance(
        step_1.id,
        str
    )

    assert step_1 == step_1.id
    assert step_1 == step_1

    assert step_2 != step_1
    assert step_2 == step_2


@pytest.fixture(
    name="migrator"
)
def make_migrator(
        step_1,
        step_2
):
    return Migrator(
        step_1,
        step_2
    )


def test_revision_steps(
        step_1,
        step_2,
        migrator
):
    revision_1, revision_2 = migrator.revisions

    assert revision_1.steps == (step_1,)
    assert revision_2.steps == (step_1, step_2)


def test_revision_ids(
        migrator
):
    revision_1, revision_2 = migrator.revisions

    assert revision_1.id != revision_2.id
    assert isinstance(
        revision_1.id,
        str
    )

    assert revision_1 != revision_2
    assert revision_1 == revision_1
    assert revision_1 == revision_1.id


def test_find_steps():
    pass
