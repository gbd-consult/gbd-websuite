Server configuration
====================

GBD WebSuite runs several server modules internally:

- the ``web`` module, which accepts and dispatches incoming requests
- the ``mapproxy`` module, which runs the bundled MapProxy and takes care of external sources, caching and reprojections
- the ``qgis`` module, which runs the bundled QGIS Server and renders qgis projects and layers
- the ``spool`` module, which handles printing and other background tasks

Each module can be disabled, if not needed (for example, if you don't use QGIS projects, there's no need to run the QGIS server). You can also configure the number of workers (roughly, CPU cores) and threads each module is allowed to use. By default, the values are ``4`` and ``0`` respectively, optimal values depend on your target machine configuration.

For high load workflows, it's also possible to run different modules on different physical machines. For example, you can set up one GBD WebSuite installation that only runs the Mapproxy module, another one for the QGIS server, and yet another one for the frontend Web module. In this case, you can specify ``host`` and ``port`` for Mapproxy and QGIS in the Web configuration, so that these can be queried over the network.


Spool server
------------

The spool module contains a *monitor*, that watches the filesystem and checks for the changes in your projects and configs and performs a hot-reload of the server if neccessary. You can configure intervals for these checks, it's recommended to set the monitor interval to at least 30 sec., because filesystem checks are resource-intensive.

