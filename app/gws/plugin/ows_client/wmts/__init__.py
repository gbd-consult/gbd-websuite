"""WMTS client.

Uses a layer of a remote WMTS service (version 1.0.0) as a tile layer.

Submodules:

- ``provider``: the WMTS service provider. Reads the capabilities, builds
  tile URL templates and fetches tiles.
- ``caps``: parser for the WMTS capabilities document. Reads the layers with
  their styles, formats and resource URLs, and the tile matrix sets.
- ``layer``: the ``wmts`` image layer that shows the tiles of one source layer.
- ``grabber``: the tile grabber used by the layer to fetch source tiles.

The layer uses a single source layer (the first selected one), a tile matrix
set and a style. The tile matrix set is the one in the forced CRS of the
provider or, by default, in the CRS that best matches the map CRS. Tile URLs
are taken from the ``ResourceURL`` template of the layer if the service
provides one, otherwise a KVP GetTile URL is built. Unless configured
otherwise, the layer extent is the extent of the first matrix, clipped to the
extent of the source layer, the resolutions are derived from the scales of the
matrices, the legend is the legend URL of the style and the metadata is the
service metadata.

The layer supports two display modes. In the default ``tile`` mode, tiles are
fetched by the server through the grabber, cached and served on the map grid,
reprojected if needed. In the ``client`` mode, the layer props include the tile
URL template and the tile matrix set, and the client loads the tiles directly
from the service; this requires a tile matrix set in the map CRS.

Reference: OGC 07-057r7, http://portal.opengeospatial.org/files/?artifact_id=35326

Example::

    map.layers+ {
        title "Base map"
        type "wmts"
        provider.url "https://example.com/wmts/1.0.0/WMTSCapabilities.xml"
        sourceLayers.names ["base_map"]
    }

    map.layers+ {
        title "Base map, loaded by the client"
        type "wmts"
        display "client"
        provider.url "https://example.com/wmts/1.0.0/WMTSCapabilities.xml"
        sourceLayers.names ["base_map"]
        withCache false
    }
"""
