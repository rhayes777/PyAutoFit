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


def test_revision_steps(
        step_1,
        step_2,
        revision_1,
        revision_2
):
    assert revision_1.steps == (step_1,)
    assert revision_2.steps == (step_1, step_2)


def test_revision_ids(
        revision_1,
        revision_2
):
    assert revision_1.id != revision_2.id
    assert isinstance(
        revision_1.id,
        str
    )

    assert revision_1 != revision_2
    assert revision_1 == revision_1
    assert revision_1 == revision_1.id


def test_difference(
        revision_1,
        revision_2,
        step_2
):
    assert (revision_2 - revision_1).steps == (step_2,)


def test_get_steps(
        migrator,
        step_1,
        step_2,
        revision_1,
        revision_2
):
    assert migrator.get_steps() == (
        step_1, step_2
    )
    assert migrator.get_steps(
        revision_1.id
    ) == (step_2,)
    assert migrator.get_steps(
        revision_2.id
    ) == ()

    assert migrator.get_steps("random") == (
        step_1, step_2
    )
