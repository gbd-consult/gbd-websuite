"""Database support.

Base classes for database connections and for the objects that read and
write database tables: layers, models and authorization providers. Concrete
implementations (for PostgreSQL/PostGIS) live in ``gws.plugin.postgres`` and
extend the classes in this package.

Submodules
----------

- ``manager`` - the database manager (``root.app.databaseMgr``). It creates
  the providers configured under ``database.providers`` and finds them by uid
  or type. A provider that fails to configure is logged and skipped. The
  manager registers itself as the ``db`` middleware.
- ``provider`` - the base database provider. It wraps an SQLAlchemy ``Engine``,
  hands out connections, reflects and caches table structures and describes
  tables and columns.
- ``connection`` - the connection object returned by ``provider.connect()``,
  a thin wrapper around an SQLAlchemy ``Connection`` with fetch helpers.
- ``model`` - the base database model, which reads features with a SELECT built
  by the model fields and creates, updates and deletes table rows.
- ``layer`` - the base vector layer for a database table, with a default model
  and finder for the table. The geometry type and CRS are taken from the table
  description. Without configured models, the layer gets a default model. If
  no extent is configured, it is computed from the table bounds. Unless
  finders are configured or search is disabled, the layer gets a default
  finder.
- ``auth_provider`` - the base authorization provider that checks users with
  SQL queries.

Providers and connections
-------------------------

Layers, models and other objects refer to a provider with ``dbUid``. Without
``dbUid`` they use the default provider passed by their parent or the first
provider of their type (see ``gws.config.util.configure_database_provider_for``).

A provider keeps one SQLAlchemy connection per thread. ``connect()`` calls can
be nested; inner calls reuse the open connection, and only the outermost one
closes it::

    with db.connect() as conn:
        rows = conn.fetch_all('SELECT * FROM my_table WHERE id = :id', id=1)
        with db.connect() as conn2:
            # the same connection
            n = conn2.fetch_int('SELECT count(*) FROM my_table')

Because the connection is shared, a ``commit()`` or ``rollback()`` on any level
ends the transaction for all levels. The ``fetch_*`` helpers roll back after
reading.

The engine is created on activation and is not pickled. Connection pooling is
off unless ``withPool`` is set; the ``pool`` options ``disabled``,
``pre_ping``, ``size``, ``recycle`` and ``timeout`` are passed to the engine.

Table structures are reflected per schema and cached for
``schemaCacheLifeTime`` seconds. ``describe`` and ``describe_column`` turn
the reflected structure into ``gws.DataSetDescription`` and
``gws.ColumnDescription`` objects, which models use to derive their fields.

Models
------

A database model builds its SELECT from the contributions of its fields
(``mc.dbSelect``), the search query (uids, keyword, shape, sort, limit, extra
conditions and columns) and the configured ``sqlFilter``. If the search asks
for uids, a keyword or a shape, but the table has no primary key or no field
contributes a matching condition, nothing is selected. Writes call the field
hooks (``before_create``, ``after_create`` and so on) around a single INSERT,
UPDATE or DELETE and commit the connection. Reads and writes check the user's
permissions on the model and raise ``gws.ForbiddenError`` if they are
missing. Composite primary keys are not supported. See ``gws.base.model`` for
the data flow between records, features and props.

SQL authorization
-----------------

The authorization provider runs two configured SELECT queries:

- ``authorizationSql`` receives the placeholders ``{username}``, ``{password}``
  and ``{token}`` from an authentication method. If it returns no rows, the
  provider does not know the user and the next provider is tried. More than one
  row is an error. Otherwise the row must contain the columns ``validuser``
  and ``validpassword`` (booleans, both must be true to log in) and ``uid``
  (the user id). The column ``roles`` is a comma-separated list of roles.
- ``getUserSql`` receives the placeholder ``{uid}`` and returns the record of
  this user, for example to restore a user from a session.

The column names ``validuser``, ``validpassword`` and ``uid`` are
case-insensitive. Other columns are passed to ``gws.base.auth.user.from_record``.
In the config file the braces of the placeholders are doubled.

Example::

    database.providers+ {
        uid "main_db"
        type "postgres"
        serviceName "LOCAL"
    }

    map.layers+ {
        type "postgres"
        dbUid "main_db"
        tableName "edit.poi"
        models+ {
            type "postgres"
            isEditable true
        }
    }

    auth.providers+ {
        type "postgres"
        dbUid "main_db"

        authorizationSql '''
            SELECT
                user.id
                    AS uid,
                user.first_name || ' ' || user.last_name
                    AS displayname,
                user.login
                    AS login,
                user.is_enabled
                    AS validuser,
                ( passwd = crypt({{password}}, passwd) )
                    AS validpassword
            FROM
                public.user
            WHERE
                user.login = {{username}}
        '''

        getUserSql '''
            SELECT
                user.id
                    AS uid,
                user.first_name || ' ' || user.last_name
                    AS displayname,
                user.login
                    AS login
            FROM
                public.user
            WHERE
                user.id = {{uid}}
        '''
    }

The ``authorizationSql`` example assumes Postgres with the ``pgcrypto`` extension.
"""

from . import provider, manager, model
