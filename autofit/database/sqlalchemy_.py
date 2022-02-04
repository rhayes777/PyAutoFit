"""
All usage of SQLAlchemy should be imported via this module to allow SQLAlchemy to be
installed optionally.

Sufficient interface is implemented to permit import of SQLAlchemy based classes without
any error. If any attempt is made to use those classes a meaningful warning is returned.
"""

try:
    import sqlalchemy as sa
    from sqlalchemy.ext import declarative
except ImportError:
    class MockSQlAlchemy:
        def __getattr__(self, item):
            return self

        def __call__(self, *args, **kwargs):
            return self

        def __mro_entries__(self, *args, **kwargs):
            return tuple()

        def declarative_base(self):
            return MockBase

        def __getitem__(self, item):
            fail()

        def __setitem__(self, key, value):
            fail()


    sa = MockSQlAlchemy()
    declarative = sa


    def fail():
        raise ImportError(
            "Please install SQLAlchemy to use the database"
        )


    class MockBase:
        def __init__(self, *args, **kwargs):
            fail()

        metadata = sa
