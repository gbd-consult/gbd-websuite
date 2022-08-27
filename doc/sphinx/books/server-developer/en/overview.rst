Overview
========

GBD WebSuite (GWS) is a server application, written in python 3.7. GWS runs in a docker container that incorporates necessary software and libraries.

Docker machine
--------------

The docker machine runs a nginx server on port 80 or 443, depending on the configuration SSL setting. The server serves static content and proxies the front web application, which is a wsgi app, running on uwsgi.

Internally, there are other servers running in the background:

- MapProxy server (port 5000), responsible for caching/reprojecting raster services. This is an uwsgi server running the MapProxy wsgi application (located in ``gws/gis/mpx``).
- QGIS server (port 4000), responsible for rendering of Qgis projects. This is an uwsgi server running the QGIS server binary.
- Spool server that runs background tasks. This is an uwsgi spooler application.

Code structure
--------------

We try to keep our python code as "typed" as possible. To achieve this we use type annotations and some code generation to create stubs for type checking. Almost every GWS source file includes the line ``import gws`` which imports generated type stubs.

Apart from type stubs, the code generator (located in ``/specgen`` and invoked as ``make spec``) creates specifications for configuration and request structures. These structures are typechecked by the server before being forwarded to specific requests handlers.

The server application is located in the ``/app`` folder in the GWS source tree. The structure of the app folder is as follows:

- ``_test`` - tests
- ``bin`` - the server startup script
- ``gws`` - main application package
- ``gws.base`` - common objects
- ``gws/config`` - configuration-related code
- ``gws/core`` - code functions and utilities
- ``gws/ext`` - server "extensions"
- ``gws/gis`` - gis related functions (e.g. projections, geometry etc)
- ``gws/qgis`` - QGIS support functions
- ``gws/server`` - server control and monitor functions
- ``gws.lib`` - tools and utilities
- ``gws/types`` - type declarations and basic structures
- ``gws/web`` - front end web application
- ``spec`` - generated config and request specifications

The "extensions" are parts of the application that provide the actual functionality. All other code might be thought of as a supporting framework. Extensions are joined in groups, currently there are following:

- ``action`` - front web server actions (request handlers)
- ``auth/provider`` - authorization providers
- ``db/provider`` - database providers
- ``helper`` - generic extensions to use by other extensions
- ``layer`` - map layer types
- ``ows/provider`` - OWS service parsers and loaders
- ``ows/service`` - OWS services
- ``search/provider`` - search providers
- ``template`` - template formats and engines

(To keep the vocabulary straight, we use the term "provider" for external things `we make use of`, and "service" for things `we make available for others`).

Configuration and the Tree
--------------------------

When the GWS server starts up, it locates and reads the configuration file. The configuration is normally a very nested structure, but at the top level it corresponds to the ``Config`` object as defined in ``common/application``. The ``Application`` object is instantiated and its ``configure`` method is invoked. This method, in turn, instantiates and configures further objects and so on, recursively. As a result, an object tree is created in memory and stays there until the server stops or reloads. Every node in the tree implements ``gws.IBaseObject`` and represents a ``common`` or an ``ext`` object. ``gws.IBaseObject`` defines a mandatory ``configure`` method, which is invoked during configuration, the ``root`` property which is the Tree's root and a set of navigation methods. Specific objects (e.g. layers) provide methods defined by their respective interfaces (e.g. ``gws.ILayer``). Some objects also provide the ``props`` getter, which exposes selected  properties to our client application.

Data and friends
----------------

All non-object structures in GWS inherit from ``gws.Data``, which is a dictionary-alike bag of values. ``gws.Data`` has some descendants for specific purposes:

- ``gws.Config`` - configuration objects
- ``gws.Request`` - request parameters
- ``gws.Response`` - request responses
- ``gws.Props`` - structures returned by ``props`` getters

All data structures used in GWS should extend one of these. Each structure must provide type annotations and, if appropriate, defaults for each member.

Coding conventions
------------------

Indents are 4 spaces, line continuations are not allowed.

Imports must be structured like this:

- system imports
- blank line
- library imports
- blank line
- ``import gws``
- import gws modules
- blank line
- ``import gws``
- blank line
- local (inter-package) imports

``from`` and ``as`` should only be used for local imports. Fully qualified names are preferred.

Members of structures available for the outside world (``Params``, ``Response``, ``Config`` and ``Props``)  should be camel case, attributes and methods of internal objects are snake case.

Comments are google-style (https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings).
