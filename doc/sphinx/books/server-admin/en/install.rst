Installation
============

Being a docker application, GBD WebSuite doesn't require any installation per se, however, you need to know a couple of things to get it running.

Directories
-----------

GBD WebSuite requires some directories to be mounted from your host machine to set the data public.

- one or more "data" directories. This is where you store your configuration and data. The server never writes to these directories, so it's a good idea to mount them readonly. These directories can be mounted to arbitrary locations in the container (we use ``/data`` by default).
- the "var" directory where the server stores its own persistent data like caches and session data. It should be mounted to ``/gws-var`` in the container.
- a temporary directory. Normally, you'd mount it as ``tmpfs``.

Ports
-----

GBD WebSuite exposes ports ``80`` and ``443``. You can map them to whatever you want during testing, and to real ``80/443`` in production.

Main configuration
------------------

GBD WebSuite expects its main configuration in ``/data/config.json``. If you prefer another location and/or format, set the environment variable ``GWS_CONFIG`` to the path of your main config.

External hosts
--------------

If your GBD WebSuite container needs external connections (most likely, to database servers), you'll need one or more ``--add-host`` options in your docker start command.

Entry point
-----------

GBD WebSuite has a single entry point, a shell script called ``gws``. To start and stop the server, use one of these ::

    gws server start
    gws server stop


Putting it all together
-----------------------

So, here are options you need to customize in your ``docker run`` command:

- one or more "data" mounts
- a "var" mount
- a "tmp" mount
- port mappings
- configuration path
- external hosts

We have a sample script ``server-sample.sh`` , which you can customize to your needs:

.. literalinclude:: /{APP_DIR}/server-sample.sh
