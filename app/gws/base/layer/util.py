import gws
import gws.base.metadata
import gws.base.model
import gws.base.search
import gws.gis.crs
import gws.gis.source
import gws.gis.zoom
import gws.lib.image
import gws.lib.style
import gws.lib.svg
import gws.types as t


class ImageFormat(t.Enum):
    """Image format"""

    png8 = 'png8'  #: png 8-bit
    png24 = 'png24'  #: png 24-bit


class ClientOptions(gws.Data):
    """Client options for a layer"""

    expanded: t.Optional[bool] = False  #: the layer is expanded in the list view
    listed: t.Optional[bool] = True  #: the layer is displayed in this list view
    selected: t.Optional[bool] = False  #: the layer is intially selected
    visible: t.Optional[bool] = True  #: the layer is intially visible
    unfolded: t.Optional[bool] = False  #: the layer is not listed, but its children are
    exclusive: t.Optional[bool] = False  #: only one of this layer's children is visible at a time


class CacheConfig(gws.Config):
    """Cache configuration"""

    enabled: bool = True
    maxAge: gws.Duration = '7d'  #: cache max. age
    maxLevel: int = 1  #: max. zoom level to cache


class Cache(gws.Data):
    maxAge: int
    maxLevel: int


class GridConfig(gws.Config):
    """Grid configuration for caches and tiled data"""

    origin: str = 'nw'  #: position of the first tile (nw or sw)
    tileSize: int = 256  #: tile size
    reqSize: int = 0  #: number of metatiles to fetch
    reqBuffer: int = 0  #: pixel buffer


class Grid(gws.Data):
    origin: str
    tileSize: int
    reqSize: int
    reqBuffer: int


class EditConfig(gws.ConfigWithAccess):
    """Edit access for a layer"""

    pass


class SearchConfig(gws.Config):
    enabled: bool = True  #: search is enabled
    providers: t.Optional[t.List[gws.ext.config.finder]]  #: search prodivers


# layer urls, handled by the map action (base/map/action.py)

_suffix = '/gws.png'


def layer_url_path(layer_uid, kind: t.Literal['box', 'tile', 'legend', 'features']) -> str:
    if kind == 'box':
        return gws.action_url_path('mapGetBox', layerUid=layer_uid) + _suffix
    if kind == 'tile':
        return gws.action_url_path('mapGetXYZ', layerUid=layer_uid) + '/z/{z}/x/{x}/y/{y}' + _suffix
    if kind == 'legend':
        return gws.action_url_path('mapGetLegend', layerUid=layer_uid) + _suffix
    if kind == 'features':
        return gws.action_url_path('mapGetFeatures', layerUid=layer_uid)


def mapproxy_layer_config(layer: gws.ILayer, mc, source_uid):
    mc.layer({
        'name': layer.uid + '_NOCACHE',
        'sources': [source_uid]
    })

    res = [r for r in layer.resolutions if r]
    if len(res) < 2:
        res = [res[0], res[0]]

    grid = getattr(layer, 'grid')

    grid.mpxUid = mc.grid(gws.compact({
        'origin': grid.origin,
        'tile_size': [grid.tileSize, grid.tileSize],
        'res': res,
        'srs': layer.bounds.crs.epsg,
        'bbox': layer.bounds.extent,
    }))

    meta_size = grid.reqSize or 4

    front_cache_config = {
        'sources': [source_uid],
        'grids': [grid.mpxUid],
        'cache': {
            'type': 'file',
            'directory_layout': 'mp'
        },
        'meta_size': [meta_size, meta_size],
        'meta_buffer': grid.reqBuffer,
        'disable_storage': not layer.hasCache,
        'minimize_meta_requests': True,
        'format': layer.imageFormat,
    }

    layer.mpxCacheUid = mc.cache(front_cache_config)

    mc.layer({
        'name': layer.uid,
        'sources': [layer.mpxCacheUid]
    })


def mapproxy_back_cache_config(layer: gws.ILayer, mc, url, grid_uid):
    source_uid = mc.source({
        'type': 'tile',
        'url': url,
        'grid': grid_uid,
        'concurrent_requests': layer.var('maxRequests', default=0)
    })

    return mc.cache(gws.compact({
        'sources': [source_uid],
        'grids': [grid_uid],
        'cache': {
            'type': 'file',
            'directory_layout': 'mp'
        },
        'disable_storage': True,
        'format': layer.imageFormat,
    }))


def generic_raster_render(layer: gws.ILayer, lri: gws.LayerRenderInput):
    if lri.type == 'box':
        return gws.LayerRenderOutput(content=generic_render_box_to_bytes(layer, lri))
    if lri.type == 'xyz':
        return gws.LayerRenderOutput(content=generic_render_xyz_to_bytes(layer, lri))


def generic_render_box_to_bytes(layer: gws.ILayer, lri: gws.LayerRenderInput):
    uid = layer.uid
    if not layer.hasCache:
        uid += '_NOCACHE'

    if not lri.view.rotation:
        return gws.gis.mpx.wms_request(uid, lri.view.bounds, lri.view.size_px[0], lri.view.size_px[1], forward=lri.extraParams)

    # rotation: render a circumsquare around the wanted extent

    circ = gws.gis.extent.circumsquare(lri.view.bounds.extent)
    w, h = lri.view.size_px
    d = gws.gis.extent.diagonal((0, 0, w, h))

    r = gws.gis.mpx.wms_request(
        uid,
        gws.Bounds(crs=lri.view.bounds.crs, extent=circ),
        width=d,
        height=d,
        forward=lri.extraParams)
    if not r:
        return

    img = gws.lib.image.from_bytes(r)

    # rotate the square (NB: PIL rotations are counter-clockwise)
    # and crop the square back to the wanted extent

    img.rotate(-lri.view.rotation).crop((
        d / 2 - w / 2,
        d / 2 - h / 2,
        d / 2 + w / 2,
        d / 2 + h / 2,
    ))

    return img.to_bytes()


def generic_render_xyz_to_bytes(layer: gws.ILayer, lri: gws.LayerRenderInput):
    grid = getattr(layer, 'grid')
    return gws.gis.mpx.wmts_request(
        layer.uid,
        lri.x,
        lri.y,
        lri.z,
        tile_matrix=grid.mpxUid,
        tile_size=grid.tileSize)
