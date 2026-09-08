"""Cache viewer: an html page to inspect cached tiles."""

import os
import re

import gws
import gws.lib.image
import gws.lib.mime

from .. import core

_DIR = os.path.dirname(__file__)
_VENDOR_DIR = f'{gws.c.APP_DIR}/gws/lib/vendor'
_ASSETS = {
    'ol.js': (f'{_VENDOR_DIR}/openlayers/ol.js', gws.lib.mime.JS),
    'ol.css': (f'{_VENDOR_DIR}/openlayers/ol.css', gws.lib.mime.CSS),
    'proj4.js': (f'{_VENDOR_DIR}/proj4/proj4.js', gws.lib.mime.JS),
    'page.js': (f'{_DIR}/page.js', gws.lib.mime.JS),
    'page.css': (f'{_DIR}/page.css', gws.lib.mime.CSS),
}
_URL = f'{gws.c.SERVER_ENDPOINT}/adminViewCache?path='
_DECOR_COLOR = (200, 0, 0, 255)


def get_content(root: gws.Root, path: str) -> gws.ContentResponse:
    path = (path or '').strip('/')

    if not path:
        return _page(root, '', -1)

    if path in _ASSETS:
        p, mime = _ASSETS[path]
        return gws.ContentResponse(contentPath=p, mime=mime)

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
    st = core.status(root, with_counts=True)
    selected = None

    if name:
        e = _find(st, name)
        z = max(z, 0)
        if z not in [lv.z for lv in e.levels]:
            raise gws.NotFoundError(f'level {z} not found')
        selected = {'name': name, 'z': z}

    config = {
        'url': _URL,
        'caches': {e.name: _cache_config(e) for e in st.entries},
        'selected': selected,
    }

    tpl = root.app.templateMgr.template_from_path(f'{_DIR}/page.cx.html')
    args = {
        'url': _URL,
        'entries': st.entries,
        'name': name,
        'z': z,
        'config': config,
    }
    return tpl.render(gws.TemplateRenderInput(args=args))


def _find(st: core.Status, name: str) -> core.Entry:
    for e in st.entries:
        if e.name == name:
            return e
    raise gws.NotFoundError(f'cache {name!r} not found')


def _tile(root: gws.Root, name: str, z: int, x: int, y: int) -> gws.ContentResponse:
    e = _find(core.status(root, cache_names=[name], with_counts=False), name)
    store = e.grabber.store
    p = store.path((x, y, z))
    if not os.path.isfile(p):
        return gws.ContentResponse(status=204, content=b'', mime=gws.lib.mime.PNG)

    img = gws.lib.image.from_path(p)
    img.add_box(_DECOR_COLOR)
    text = f'{p[len(store.baseDir) + 1 :]}\n{os.path.getsize(p) / 1024:.1f}K'
    draw = gws.lib.image.get_draw(img)
    font = gws.lib.image.get_font(11)
    x0, y0, x1, y1 = draw.multiline_textbbox((4, 3), text, font=font)
    draw.rectangle((x0 - 2, y0 - 1, x1 + 2, y1 + 1), fill=(255, 255, 255, 200))
    draw.multiline_text((4, 3), text, font=font, fill=_DECOR_COLOR)
    return gws.ContentResponse(content=img.to_bytes(gws.lib.mime.PNG), mime=gws.lib.mime.PNG)


def _cache_config(e: core.Entry) -> dict:
    gr = e.grabber
    crs = gr.targetCrs
    return {
        'crs': {
            'epsg': crs.epsg,
            'proj4text': crs.proj4text,
            'extent': list(crs.extent),
        },
        'gridExtent': list(gr.grid.extent),
        'tileSize': gr.grid.tileSize,
        'extent': list(gr.extent),
        'ext': gr.store.extension,
        'resolutions': {lv.z: lv.resolution for lv in e.levels},
        'ranges': {lv.z: list(lv.cachedRange) if lv.cachedRange else None for lv in e.levels},
        'counts': {lv.z: [lv.totalTiles, lv.cachedTiles] for lv in e.levels},
        'srid': crs.srid,
    }
