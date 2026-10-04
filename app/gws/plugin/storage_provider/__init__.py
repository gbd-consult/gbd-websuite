"""Storage providers.

Storage providers keep the named entries that users save with client tools,
for example selections, annotations or dimensions. They implement the
``gws.StorageProvider`` interface and are configured in the ``storage``
section of the application; the storage objects in ``gws.base.storage``
use them.

Subpackages
-----------

- ``sqlite`` - the ``sqlite`` provider, which stores entries in an SQLite
  database file.

Example::

    storage {
        providers+ {
            type "sqlite"
            path "/data/storage.sqlite"
        }
    }
"""
