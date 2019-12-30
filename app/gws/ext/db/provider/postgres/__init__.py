import gws.types as t
from .provider import Object


class Config(t.WithType):
    """Postgres/Postgis database provider"""

    database: str = ''  #: database name
    host: str = 'localhost'  #: database host
    password: str  #: password
    port: int = 5432  #: database port
    timeout: t.Duration = 0  #: query timeout
    uid: str  #: unique id
    user: str  #: username
