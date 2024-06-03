"""OWS services.

OGC Standards:

- OpenGIS Web Map Service (WMS) Implementation Specification 1.3.0 06-042
    https://portal.ogc.org/files/?artifact_id=14416

- Web Map Service 1.1.1 01-068r3
    https://portal.ogc.org/files/?artifact_id=1081&format=pdf

- OpenGIS Web Map Tile Service Implementation Standard 1.0.0 07-057r7
    https://portal.ogc.org/files/?artifact_id=35326

- OpenGIS Web Feature Service 2.0 Interface Standard (also ISO 19142) 2.0 09-025r1
    https://portal.ogc.org/files/?artifact_id=39967

- OpenGIS Web Feature Service (WFS) Implementation Specification 1.1.0 04-094
    https://portal.ogc.org/files/?artifact_id=8339

- OGC® Web Coverage Service (WCS) Interface Standard – Core, version 2.1 17-089r1
    https://portal.opengeospatial.org/files/17-089r1

- OGC Web Service Common Implementation Specification 2.0.0 06-121r9
    https://portal.ogc.org/files/?artifact_id=38867

Other implementations:

- https://mapserver.org/ogc/wms_server.html
- https://docs.geoserver.org/latest/en/user/services/wms/reference.html
- https://mapserver.org/ogc/wfs_server.html
- https://docs.geoserver.org/latest/en/user/services/wfs/reference.html
- https://mapserver.org/ogc/wcs_server.html
- https://docs.geoserver.org/latest/en/user/services/wcs/reference.html
"""

from .core import (
    LayerCaps,
    FeatureCollection,
    FeatureCollectionMember,
)
from . import service, layer_caps, request, error
from .request import TemplateArgs
