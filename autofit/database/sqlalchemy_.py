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


    sa = MockSQlAlchemy()
    declarative = MockSQlAlchemy()
