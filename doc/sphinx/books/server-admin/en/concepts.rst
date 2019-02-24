Basic concepts
==============

Requests and urls
-----------------

Once launched, GBD WebSuite listens on ports ``80/443`` and processes ``GET`` and ``POST`` requests. Like a conventional webserver, GWS can serve static content, like html pages or images, but its main purpose is to provide dynamic map imagery and data. For dynamic requests, there's a single endpoint (url), namely the ``_`` (underscore). All requests to this endpoint are expected to contain the command (``cmd``) parameter.
Additionally, all ``POST`` requests must be in JSON format.

Here a few examples of requests GBD WebSuite can handle:

Staic web requests ::

    http://maps.my-server.com/images/smile.jpg

Dynamic GET requests (e.g. map imagery) ::

    http://maps.my-server.com/_?cmd=mapHttpGetBbox&layer=london.metro&width=100&height=200&bbox=10,20,30,40

Dynamic POST request (e.g. search) ::

    {
        "cmd":"searchRun",
        "params": {
            "projectUid": "london",
            "bbox": [10,20,30,40],
            "keyword": "queen"
        }
    }

Dynamic GET urls can be modified using URL rewriting, so that this ::

    http://maps.my-server.com/wms/london/metro

can be used instead of ::

    http://maps.my-server.com/_?cmd=wmsHttpGetMap&project=london&layers=metro


Sites and projects
------------------

On the top level, GWS works with two types of entities: *projects* and *sites*. A project is roughly a map and a collection of settings related to that specific map. A site is a domain name, bound to authorization and routing rules.

In the above example, ``london`` is a project, ``metro`` is a layer configured for that project, while the domain name ``maps.my-server.com`` and the corresponding rewrite rule are taken from the site configuration.

Sites and projects are orthogonal concepts, and you can run the same project under multiple sites. For example, if you decide to change ``maps.my-server.com`` to e.g. ``gis.my-other-server.com`` this wouldn't require any changes in the project ``london``.


Actions
-------

The set of commands (``cmd`` in the above examples) is not fixed and is freely configurable. Commands are grouped into *actions*, you can configure available actions globally or on a per-project basis.


Maps, layers and sources
------------------------

Every GBD WebSuite projects contains at least one *map*, which is a collection of *layers*. There are different types of layers (for example, "wms", or "tile"). You can configure access rights, view properties (like an extent) and the metadata for the whole map and for each layer individually. A layer configuration typically contains instructions for the server how to transform the source data, for example

- reproject the data
- convert the imagery from WMS into tiles and vice versa
- reformat feature data
- apply custom styles to features


Pluggable architecture
----------------------

Almost all GBD WebSuite features are implemented as plugins. We have plugins for these types of objects

TABLE
   actions ~ Server actions
   authorization  providers ~ Handle authorization and authentication
   database providers ~ Database connections
   search provides ~ Handle fulltext and attribute search
   layers ~ Map layers
   sources ~ Geodata sources for maps
   print templates ~ Various print template formats
/TABLE

Pluggable objects in the configuration are identifed by their ``type`` property.


Configuration files and objects
-------------------------------

GBD WebSuite supports several configuration formats:

- json, in which case the config file name must end with ``config.json``
- yaml (``config.yaml``). We use json in these docs, but you can always use yaml with the same structure if you like it more
- python (``config.py``). Complex, repetitve or highly dynamic configurations can be also written in straight python. Your python config script must contain a function called ``config()`` returning a ``dict`` with the same structure as JSON. Note that your config module is executed inside the container, so it must be compatible with Python 3.6.

Configuration starts with the main config file (``GWS_CONFIG``), which can include other config files for projects and project groups. Once all files are read and parsed, all configured objects are assembled into a large "tree", with the ``Application`` object being the root node. Here's an example of such a tree ::


    Application
    |
    |-- auth options
    |-- server options
    |-- web options
    |
    \-- projects
        |
        |-- First project
        |   |-- project options
        |   \-- Map
        |       |-- First layer
        |       \-- Second layer
        |
        \-- Second project
           |-- project options
           \-- Map
               \-- Layer group
                   \-- Sub-layer


Most configuration options are inheritable, that means, when the system looks for some property for a layer, and it's not configured explicitly, then the parent layer is consulted, then the map, then the containing project and finally the root ``Application``.
