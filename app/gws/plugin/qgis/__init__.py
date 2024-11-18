"""QGIS support.

The QGIS plugin supports parsing and writing QGIS projects, stored in paths or in a Postgres database.
The parsing is based on directly reading project XML files; no QGIS APIs are used.
The projects are parsed into capabilities objects, from which QGIS-based layers and finders can be created.
The rendering of QGIS layers is achieved by sending requests to a QGIS Server instance.


Project and layer extents
-------------------------

QGIS does not provide a way to obtain complete project and layer extents, with respect to symbology.
Only data-based extents are known at parse time. We compute extents with the following logic:

- if a project provides an explicit WMS extent (Project Properties->QGIS Server->WMS), this extent is used as project render extent (`bounds`)
- otherwise, if `useCanvasExtent` is true, the canvas extent is used
- otherwise, the project render extent is set to the union of layers' data extents + the configured `extentBuffer`
- if the render extent is empty, the CRS extent is taken
- for layers, the data extent is either an explicit extent (Layer Properties->Metadata->Extent) or an implicit data extent
- the layer data extent is used as a "zoom" extent, but when rendering a layer, the project extent is used
"""

from . import provider, project, caps
