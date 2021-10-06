import pickle

from autofit.messages.transform import log_10_transform


def test_pickle_log_10_transform():
    loaded = pickle.loads(
        pickle.dumps(
            log_10_transform
        )
    )
    assert loaded.transform(10) == 1
