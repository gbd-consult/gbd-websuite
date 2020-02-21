Storage
=======

In the GBD WebSuite some elements, like Annotations or Dimensions, can be saved for later reuse. The server provides a simple virtual file system, or storage. To use the storage, the ``storage`` action and the ``storage`` helper must be configured.

The storage is divided into "categories", corresponding to the client element type ("Annotation", "Dimension" etc). Specific client elements can be saved under user-choosen names in a respective category.

Storage is currently backed up by a local sqlite database, in the future we'll add a postgres-based backend.


Permissions
-----------

You can configure read, write or read-write access to specific categories, or to all categories for specific user roles.

In this example configuration, the "users" role is granted full access to "Dimension" and read access to "Annotate", and the "administrators" role is granted full access  to all categories ::

    "helpers": {
        "storage": {
            "backend": "sqlite",
            "permissions": [
                {
                    "category": "Dimension",
                    "mode": "all",
                    "access": {
                        "role": "users",
                        "type": "allow"
                    }
                },
                {
                    "category": "Annotate",
                    "mode": "read",
                    "access": {
                        "role": "users",
                        "type": "allow"
                    }
                },
                {
                    "category": "*",
                    "mode": "all",
                    "access": {
                        "role": "administrators",
                        "type": "allow"
                    }
                }
            ]
        }
    }

