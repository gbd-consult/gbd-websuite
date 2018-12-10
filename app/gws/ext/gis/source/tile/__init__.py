import re

import gws.config
import gws.gis.proj
import gws.gis.source
import gws.types as t


class Config(gws.gis.source.BaseConfig):
    """tiled source"""

    grid: t.Optional[t.GridConfig]  #: source grid configuration
    url: str  #: source url, with {x}, {y} and {z} placeholders


class Object(gws.gis.source.Base, t.SourceObject):
    def service_metadata(self):
        return t.ServiceMetaData()

    def layer_metadata(self, layer_name: str):
        return t.MetaData()

    def mapproxy_config(self, mc, options=None):
        # for tiled sources, we use the 'double cache' technique as described here:
        # https://mapproxy.org/docs/1.11.0/configuration_examples.html#reprojecting-tiles
        # basically, we configure a source with an IN no-store cache and then give it to layer.mapproxy_config
        # which attaches an OUT cache to it

        # we use {x} like in Ol, mapproxy want %(x)s
        url = re.sub(
            r'{([xyz])}',
            r'%(\1)s',
            self.var('url'))

        res = self.var('resolutions', parent=True)

        grid = self.var('grid', parent=True)
        src_crs = self.var('crs', parent=True)

        src_grid_config = gws.compact({
            'origin': grid.origin,
            'res': res,
            'srs': src_crs,
            'tile_size': [grid.tileSize, grid.tileSize],
        })

        # if the source has its own extent, it must be in the source's crs
        # otherwise, we expect crs to be 3857 and rely on mapproxy defaults
        # if crs is not 3857, we get an error from MP
        # if self.view.extent:
        #     src_grid_config['bbox'] = self.view.extent
        #     src_grid_config['bbox_srs'] = src_crs

        # src_grid_config = gws.extend(src_grid_config, grid.options)

        if 'bbox' not in src_grid_config:
            # @TODO projections other than mercator require a bbox
            pass

        src_grid = mc.grid(self, src_grid_config, self.uid + '_src')

        source = mc.source(self, {
            'type': 'tile',
            'url': url,
            'grid': src_grid,
        })

        src_cache_options = {
            'type': 'file',
            'directory_layout': 'mp'
        }

        # NB for tiled sources always request one tile at a time

        src_cache_config = gws.compact({
            'sources': [source],
            'grids': [src_grid],
            'cache': src_cache_options,
            'disable_storage': True,
            'meta_size': [1, 1],
            'meta_buffer': 0,
            'minimize_meta_requests': True,
        })

        src_cache = mc.cache(self, src_cache_config, self.uid + '_src')
        return src_cache
