"""Database connection wrapper."""

import gws
import gws.lib.sa as sa


class Object(gws.DatabaseConnection):
    """Database connection."""

    db: gws.DatabaseProvider
    """Provider this connection belongs to."""
    saConn: sa.Connection
    """The underlying SQLAlchemy connection."""

    def __init__(self, db: gws.DatabaseProvider, conn: sa.Connection):
        """Create a connection wrapper.

        Args:
            db: Provider this connection belongs to.
            conn: The SQLAlchemy connection.
        """
        self.db = db
        self.saConn = conn

    def execute(self, stmt, params=None, execution_options=None):
        if isinstance(stmt, str):
            stmt = sa.text(stmt)
        return self.saConn.execute(stmt, params, execution_options=execution_options)

    def fetch_all(self, stmt, **params):
        return [r._asdict() for r in self.execute(stmt, params)]

    def fetch_first(self, stmt, **params):
        res = self.execute(stmt, params)
        r = res.first()
        return r._asdict() if r else None

    def fetch_scalars(self, stmt, **params):
        res = self.execute(stmt, params)
        return list(res.scalars().all())

    def fetch_strings(self, stmt, **params):
        res = self.execute(stmt, params)
        return [_to_str(s) for s in res.scalars().all()]

    def fetch_ints(self, stmt, **params):
        res = self.execute(stmt, params)
        return [_to_int(s) for s in res.scalars().all()]

    def fetch_scalar(self, stmt, **params):
        res = self.execute(stmt, params)
        return res.scalar()

    def fetch_string(self, stmt, **params):
        res = self.execute(stmt, params)
        s = res.scalar()
        return _to_str(s) if s is not None else None

    def fetch_int(self, stmt, **params):
        res = self.execute(stmt, params)
        s = res.scalar()
        return _to_int(s) if s is not None else None


##


def _to_int(s) -> int:
    """Return the value if it is an int, raise ``ValueError`` otherwise."""
    if isinstance(s, int):
        return s
    raise ValueError(f'db: expected int, got {s=}')


def _to_str(s) -> str:
    """Convert a value to a string, ``None`` to an empty string."""
    if s is None:
        return ''
    return str(s)
