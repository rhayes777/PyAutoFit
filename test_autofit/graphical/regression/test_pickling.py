import pickle

from autofit.messages import UniformNormalMessage
from autofit.messages.transform import log_10_transform
from autofit.messages.transform_wrapper import TransformedWrapperInstance


def test_pickle_log_10_transform():
    loaded = pickle.loads(
        pickle.dumps(
            log_10_transform
        )
    )
    assert loaded.transform(10) == 1


def test_copy_and_pickle():
    original = UniformNormalMessage(
        mean=1.0,
        sigma=2.0,
    )
    copied = original.copy()
    assert isinstance(
        copied,
        TransformedWrapperInstance
    )
    assert copied is not original
