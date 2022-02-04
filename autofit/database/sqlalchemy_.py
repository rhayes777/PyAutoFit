try:
    import sqlalchemy as sa
    from sqlalchemy.ext import declarative
except ImportError:
    class MockSQlAlchemy:
        pass


    sa = MockSQlAlchemy()
    declarative = MockSQlAlchemy()
