import re

import gws
import gws.gis.layer
import gws.types as t
import gws.gis.source
import gws.tools.misc

_EPSG_3857_EXTENT = [
    -20037508.34,
    -20037508.34,
    20037508.34,
    20037508.34
]


class ServiceConfig:
    """Tile service configuration"""

    extent: t.Optional[t.Extent]  #: service extent
    crs: t.crsref = 'EPSG:3857'  #: service CRS
    origin: str = 'nw'  #: position of the first tile (nw or sw)
    tileSize: int = 256  #: tile size


class Config(gws.gis.layer.ProxiedConfig):
    """Tile layer"""

    display: str = 'tile'
    service: t.Optional[ServiceConfig] = {}  #: service information
    url: t.url  #: rest url with placeholders {x}, {y} and {z}


class Object(gws.gis.layer.ProxiedTile):
    def __init__(self):
        super().__init__()

        self.service: ServiceConfig = None
        self.url = ''

    def configure(self):
        super().configure()

        self.url = self.var('url')
        self.service = self.var('service')

        if not self.service.extent:
            if self.service.crs == 'EPSG:3857':
                self.service.extent = _EPSG_3857_EXTENT
            else:
                raise gws.Error(r'service extent required for crs {self.service.crs!r}')

        self.cache_uid = gws.tools.misc.sha256(self.url)

    def mapproxy_config(self, mc, options=None):
        # we use {x} like in Ol, mapproxy wants %(x)s
        url = re.sub(
            r'{([xyz])}',
            r'%(\1)s',
            self.url)

        src_grid_config = gws.compact({
            'origin': self.service.origin,
            'bbox': self.service.extent,
            # 'res': res,
            'srs': self.service.crs,
            'tile_size': [self.service.tileSize, self.service.tileSize],
        })

        src = self.mapproxy_back_cache_config(mc, url, src_grid_config)
        self.mapproxy_layer_config(mc, src)
