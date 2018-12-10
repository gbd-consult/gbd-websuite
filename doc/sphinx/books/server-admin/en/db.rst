Databases
=========

GBD WebSuite can fetch geodata from databases, either indirectly, using QGIS and WMS/WMTS providers, or by connecting to a database directly. For the latter case, you need to configure database connections in the main application config.

At the moment we only support PostgreSQL/PostGIS databases. We plan to add Sqlite/SpatiaLite, MySQL and MongoDB later on.

Example of a PostGIS provider configuration ::

    ## in the main config:

    "db": {
        "providers" [
            {
                "type": "postgis",
                "uid": "my_db_connection",
                "host": "DBHOST",
                "port": 5432,
                "database": "mydatabase",
                "user": "me",
                "password": "secret"
            }
        ]
    }

If you have multiple credentials to the same server, you have to configure them as different providers.

Somewhere else in your configuration, when a database connection is required, you simply mention the provider unique id. Example for a SQL search config ::

        ## search configuration

        {
            "type": "sql",
            "db": "my_db_connection",
            ...other options
        }

Since GBD WebSuite runs in a container, you cannot use ``localhost`` as a hostname even if your DB server runs on the same physical machine. Instead, the docker host IP address, like ``172.17.0.1`` should be used (the exact value depends on your docker network settings). For portability reasons, it's recommended to alias it with ``--add-host``.
