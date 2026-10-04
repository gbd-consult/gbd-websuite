"""PostgreSQL/PostGIS support.

Provides a database provider for PostgreSQL and the objects that use it:
layers, models, finders, an authorization provider and a storage provider.
The database-independent parts are in ``gws.base.database``; this plugin adds
the PostgreSQL specifics.

Submodules:

- ``provider``: the ``postgres`` database provider. Builds the connection URL
  from host and credentials or from a service name in the PostgreSQL service
  file, splits and joins schema-qualified table names (the default schema is
  ``public``; quoted identifiers are supported), computes table bounds with
  ``ST_Extent`` and detects array columns (described as ``strlist``,
  ``intlist`` or ``floatlist`` when the item type is known) and the geometry
  type and SRID of PostGIS columns, also for constraint-based geometry columns
  without a type modifier, via ``public.geometry_columns``.
- ``layer``: the ``postgres`` vector layer that shows the features of a table.
- ``model``: the ``postgres`` model that reads and writes the records of a table.
- ``finder``: the ``postgres`` finder that searches a table by keyword,
  geometry and filter. The search itself is done by the ``postgres`` models of
  the finder; all three search types are always enabled, since the actual
  support depends on the models.
- ``auth_provider``: the ``postgres`` authorization provider that checks
  credentials and loads users with SQL queries.
- ``storage_provider``: the ``postgres`` storage provider that keeps saved
  user data in a table. The table is not created by the provider; writing a
  record with an existing category and name replaces it.

Objects find their database provider by ``dbUid``. Without ``dbUid``, the
first configured ``postgres`` provider is used.

Example::

    database.providers+ {
        uid "main_db"
        type "postgres"
        serviceName "LOCAL"
    }

    map.layers+ {
        title "POIs"
        type "postgres"
        dbUid "main_db"
        tableName "edit.poi"
    }

    storage.providers+ {
        type "postgres"
        dbUid "main_db"
        tableName "public.gws_storage"
    }
"""
