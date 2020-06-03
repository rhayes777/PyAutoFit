import pytest

from autofit import message_passing as mp


@pytest.fixture(
    name="x"
)
def make_x():
    return mp.Variable("x")
