import pytest


def test_already_compared(
        aggregator
):
    with pytest.raises(
            AssertionError
    ):
        print((aggregator.centre == 1) == 1)

    with pytest.raises(
            AssertionError
    ):
        print((aggregator.centre == 1).intesity)
