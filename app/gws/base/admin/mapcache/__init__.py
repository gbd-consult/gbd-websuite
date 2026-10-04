"""Cache viewer: an html page to inspect cached tiles.

The page lists all tile caches from the cache inventory (``gws.gis.cache.core``)
with per-level statistics, and shows the cached tiles of a selected cache and
level on an OpenLayers map. It is served by the ``adminMapCache`` command of the
``admin`` action.

Paths:

- empty: the overview page.
- ``<cache>``, ``<cache>/<z>``: the page with a cache and level selected (level 0 by default).
- ``<cache>/<z>/<x>/<y>.<ext>``: a single cached tile as PNG, or an empty 204 response if
  the tile is not cached.
- ``ol.js``, ``ol.css``, ``proj4.js``, ``page.js``, ``page.css``: page assets.

Example::

    /_/adminMapCache?path=
    /_/adminMapCache?path=<cache name>/12
"""

import os
import re

import gws
import gws.lib.crs
import gws.lib.grid
import gws.lib.image
import gws.lib.mime
import gws.gis.cache.core as core

_DIR = os.path.dirname(__file__)
_VENDOR_DIR = f'{gws.c.APP_DIR}/gws/lib/vendor'
_ASSETS = {
    'ol.js': (f'{_VENDOR_DIR}/openlayers/ol.js', gws.lib.mime.JS),
    'ol.css': (f'{_VENDOR_DIR}/openlayers/ol.css', gws.lib.mime.CSS),
    'proj4.js': (f'{_VENDOR_DIR}/proj4/proj4.js', gws.lib.mime.JS),
    'page.js': (f'{_DIR}/page.js', gws.lib.mime.JS),
    'page.css': (f'{_DIR}/page.css', gws.lib.mime.CSS),
}
_URL = f'{gws.c.SERVER_ENDPOINT}/adminMapCache?path='
_DECOR_COLOR = (200, 0, 0, 255)


def get_content(root: gws.Root, path: str) -> gws.ContentResponse:
    """Render the cache viewer page, a cached tile or a page asset.

    Args:
        root: The configuration root.
        path: Page, tile or asset path.

    Returns:
        The rendered page, the tile image or the asset.

    Raises:
        gws.NotFoundError: If the path is invalid or the cache or level is not found.
    """
    path = (path or '').strip('/')

    if not path:
        return _page(root, '', -1)

    if path in _ASSETS:
        p, mime_type = _ASSETS[path]
        return gws.ContentResponse(contentPath=p, mimeType=mime_type)

    m = re.match(r'^(\w+)/(\d+)/(\d+)/(\d+)\.\w+$', path)
    if m:
        name, z, x, y = m.groups()
        return _tile(root, name, int(z), int(x), int(y))

    m = re.match(r'^(\w+)$', path)
    if m:
        return _page(root, m.group(1), -1)

    m = re.match(r'^(\w+)/(\d+)$', path)
    if m:
        return _page(root, m.group(1), int(m.group(2)))

    raise gws.NotFoundError(f'invalid path {path!r}')


##


def _page(root: gws.Root, name: str, z: int) -> gws.ContentResponse:
    """Render the page, with a cache and level selected if ``name`` is given."""
    inv = core.inventory(root)
    core.add_stats(inv)
    selected = None

    if name:
        c = _find(inv, name)
        z = max(z, 0)
        if z not in [lv.z for lv in c.levels]:
            raise gws.NotFoundError(f'level {z} not found')
        selected = {'name': name, 'z': z}

    config = {
        'url': _URL,
        'caches': {c.name: _cache_config(c) for c in inv.caches},
        'selected': selected,
        'baseGrid': _grid_config(gws.lib.grid.for_crs(gws.lib.crs.WEBMERCATOR)),
    }

    tpl = root.app.templateMgr.template_from_path(f'{_DIR}/page.cx.html')
    args = {
        'url': _URL,
        'caches': inv.caches,
        'name': name,
        'z': z,
        'config': config,
    }
    return tpl.render(gws.TemplateRenderInput(args=args))


def _find(inv: core.Inventory, name: str) -> core.Cache:
    """Find a cache in the inventory by name, raise ``NotFoundError`` if not found."""
    for c in inv.caches:
        if c.name == name:
            return c
    raise gws.NotFoundError(f'cache {name!r} not found')


def _tile(root: gws.Root, name: str, z: int, x: int, y: int) -> gws.ContentResponse:
    """Return a cached tile as PNG, or an empty 204 response if it is not in the store."""
    c = _find(core.inventory(root), name)
    store = c.grabber.store
    p = store.path((x, y, z))
    if not os.path.isfile(p):
        return gws.ContentResponse(status=204, content=b'', mimeType=gws.lib.mime.PNG)

    img = gws.lib.image.from_path(p)
    
    # img.add_box(_DECOR_COLOR)
    # text = f'{p[len(store.baseDir) + 1 :]}\n{os.path.getsize(p) / 1024:.1f}K'
    # draw = gws.lib.image.get_draw(img)
    # font = gws.lib.image.get_font(11)
    # x0, y0, x1, y1 = draw.multiline_textbbox((4, 3), text, font=font)
    # draw.rectangle((x0 - 2, y0 - 1, x1 + 2, y1 + 1), fill=(255, 255, 255, 200))
    # draw.multiline_text((4, 3), text, font=font, fill=_DECOR_COLOR)
    
    return gws.ContentResponse(content=img.to_bytes(gws.lib.mime.PNG), mimeType=gws.lib.mime.PNG)


def _cache_config(c: core.Cache) -> dict:
    """Return the client-side configuration of a cache: CRS, grid, extent and per-level statistics."""
    gr = c.grabber
    crs = gr.targetCrs
    return {
        'crs': {
            'epsg': crs.epsg,
            'proj4text': crs.proj4text,
        },
        'gridExtent': list(gr.grid.extent),
        'tileSize': gr.grid.tileSize,
        'extent': list(gr.extent),
        'ext': gr.store.extension,
        'resolutions': {lv.z: lv.resolution for lv in c.levels},
        'ranges': {lv.z: list(lv.cachedRange) if lv.cachedRange else None for lv in c.levels},
        'gridRanges': {lv.z: list(lv.gridRange) for lv in c.levels},
        'counts': {lv.z: [lv.totalTiles, lv.cachedTiles] for lv in c.levels},
        'srid': crs.srid,
        'grid': _grid_config(gr.grid),
    }


def _grid_config(mg: gws.MapGrid) -> dict:
    """Return the client-side configuration of a grid."""
    return {
        'extent': list(mg.extent),
        'baseResolution': mg.baseResolution,
        'tileSize': mg.tileSize,
    }
