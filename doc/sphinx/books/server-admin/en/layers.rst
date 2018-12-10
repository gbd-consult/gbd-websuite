Layers
======

A *layer* on a GBD WebSuite project is identified by its ``type``, in addition, layers have the following properties (if not explcitly configured, there will be inherited from the parent layer or from the map):

* ``source`` - where the layer gets its geodata (see :doc:`sources`)
* ``view`` - spatial properties of the layer (extent, set of allowed resolutions or scales for this layer)
* ``cache`` and ``grid`` - affect the layer caching (see :doc:`cache`)
* ``clientOptions`` - options for the GBD WebSuite client (see :doc:`client`)
* ``attributes`` - layer meta-data (e.g. attribution)
* ``meta`` - transformation rules for features (see :doc:`features`)

Layer types
-----------

box
~~~

A box layer is similar to a conventional WMS layer. It's queried with WMS parameters ``bbox``, ``width`` and ``height`` and returns a ``png`` image.

tile
~~~~

A tile layer works as an XYZ tiled source. Note that, in exception to the general rule, requests to tile layers imitate static requests, to allow client-side caching. An example of tile layer request ::

    http://example.org/_/cmd/mapHttpGetXyz/layer/project.layer/z/1/x/2/y/3/t.png


group
~~~~~

Group layers contain other layers, they don't provide any geodata by themselves. Apart from visual grouping, another purpose of a group is to maintain access or fallback Cache and Grid configurations for its child layers. A group can be made "virtual", or ``unfolded``, in which case it's not displayed in the client, while its child layers are.

tree
~~~~

A tree layer is capable of diplaying a whole hierarchy of layers from a WMS or QGIS source. A tree layer will be displayed as a Group in the client and with source layers as its child nodes (or *leaves*).

It's also possible to select only specific layers from the source. When reading the source, the server creates a  virtual *path* property for each layer, which contains the layer unique id and its parent ids, similar to filesystem paths, like ``/root-layer-id/grandparent-id/parent-id/layer-id``. The ``pathMatch`` regex can be used to filter layers with matching paths.

qgis
~~~~

QGIS layers are similar to tree layers, but they only work with QGIS maps. Instead of a single ``pathMatch``, they can have a list of match rules, which tell the server how to handle matching QGIS layers. For example, you can make "tilify" a particular layer, or "flatten" a certain subtree into a single layer.

vector
~~~~~~

Vector layers are rendered on the client. When a vector layer is requested, the server sends the GeoJSON list of features and style description to the client, which is then supposed to perfrom the actual rendering.
