Remote administration
=====================

GBD WebSuite can be administrated remotely, from a local machine. To use remote administration, you need client-side software, like our QGIS Plugin and some server configuration.

To configure remote administration server-side, you need to enable an action (see :doc:`actions`) of type ``remoteadmin`` under ``api/actions`` ::

    "api": {
        "actions":
            ...
            {
                "type": "remoteadmin"
            }


Remote administration skips normal authorization mechanisms and uses the password, which is stored encrypted in the file  ``.remoteadmin`` in the server ``gws-var`` directory (see :doc:`install`). You can change the default path the ``passwordFile`` option. To create the password file, use the command ``auth passwd`` (see :doc:`cli`) ::

    gws auth passwd --path /gws-var/.remoteadmin

