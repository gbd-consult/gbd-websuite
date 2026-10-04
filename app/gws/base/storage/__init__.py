"""Storage for data saved by users.

Some client tools let users save their work under a name and load it later, for
example annotations or dimensions. This package provides the server side for that.

Submodules:

- ``manager``: the storage manager (``gws.StorageManager``), configured in the
  ``storage`` section of the application. It holds the storage providers
  (``gws.StorageProvider``), which store named entries per category. Providers
  are plugins, for example ``sqlite``. They are created as shared objects; a
  provider with the same uid as an existing one replaces it.
- ``core``: the storage object, a node that an action creates for its client
  tool. It binds one provider and one category, checks the user's permissions
  and handles the storage requests (``list``, ``read``, ``write``, ``delete``) of
  the client. The ``Request``, ``Response`` and ``Props`` types are used by the
  actions to define their storage command.

Access to the entries is controlled by the ``access`` of the storage object:
``read`` allows to list and read entries, ``write`` to overwrite existing ones,
``create`` to add new ones and ``delete`` to delete them. Without a ``providerUid``
the first configured provider is used.

Example::

    {
        storage {
            providers [
                { type "sqlite" }
            ]
        }
        actions [
            {
                type "dimension"
                storage { access "allow all" }
            }
        ]
    }

Example::

    class Object(gws.base.action.Object):
        def configure(self):
            self.storage = self.create_child_if_configured(
                gws.base.storage.Object, self.cfg('storage'), categoryName='MyTool')

        @gws.ext.command.api('myToolStorage')
        def handle_storage(self, req, p: gws.base.storage.Request) -> gws.base.storage.Response:
            return self.storage.handle_request(req, p)
"""

from .core import Request, Response, Config, Props, Object
from . import manager
