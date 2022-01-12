import datetime as dt


def test_import():
    """
    Ensure that import time is less than 100 microseconds
    """
    tic = dt.datetime.now()
    # noinspection PyUnresolvedReferences
    import autofit
    difference = (dt.datetime.now() - tic).microseconds
    assert difference < 100
