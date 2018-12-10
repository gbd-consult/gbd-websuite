Caching framework
=================


The bundled Mapproxy server can be configured to cache geodata from external WMS, WMTS or KVP sources. In GBD WebSuite, you use ``Cache`` and ``Grid`` options in the Map config, or on a per-layer basis, and the ``Seeding`` configuration in the main config.

There are a few steps neccessary to make a layer cacheable:

* the layer must have a defined ``View`` with an ``extent`` and a set of allowed ``resolutions`` or ``scales``. These settings can be defined in the Layer config or inherited from the Map config

* the layer (or the map) must have a ``Grid``. For WMS sources, it's important to set up the meta-tiling correctly, to avoid the "dangling labels" problem (see https://mapproxy.org/docs/latest/labeling.html)

* the layer (or the map) must have a ``Cache`` with ``enabled`` set to ``true``

Once the caching is set up, it's automatically populated when users browse your maps. You can also pre-seed the cache, using ``gws cache`` command line tools.

Important: if you change View or Grid configurations, you have to remove the cache for the layer or the map to prevent unpleasant artifacts.
