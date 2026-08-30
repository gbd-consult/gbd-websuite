"""Mockmap services: protocols, tile addressing, rendering."""

import math
import os
import threading

import gws
import gws.lib.crs
import gws.lib.image
import gws.lib.vendor.jump

from . import scene

METERS_PER_DEGREE = 111319.4907932736
PIXEL_SIZE = 0.00028

_cache_lock = threading.Lock()
_cache: dict[tuple, bytes] = {}



class Tms:
    def __init__(self, cfg: dict, crs: gws.Crs, origin: str):
        self.crs = crs
        self.uid = f'EPSG_{crs.srid}'
        self.extent = tuple(cfg['tmsExtent'])
        self.tileSize = cfg.get('tileSize') or 256
        self.resolutions = [float(r) for r in cfg['resolutions']]
        self.origin = cfg.get('origin') or origin

        self.bbox = _bbox(self.extent)
        self.originX = self.extent[0]
        self.originY = self.extent[3] if self.origin == 'nw' else self.extent[1]

        mpu = METERS_PER_DEGREE if self.crs.isGeographic else 1.0
        tlc = (self.extent[0], self.extent[3])
        if self.crs.isYX:
            tlc = (self.extent[3], self.extent[0])

        self.matrices = []
        for z, res in enumerate(self.resolutions):
            w, h = self.matrix_size(z)
            self.matrices.append(gws.Data(
                z=z,
                resolution=res,
                scaleDenominator=res * mpu / PIXEL_SIZE,
                topLeftCorner=f'{tlc[0]} {tlc[1]}',
                matrixWidth=w,
                matrixHeight=h,
            ))

    def span(self, z: int) -> float:
        return self.tileSize * self.resolutions[z]

    def matrix_size(self, z: int) -> tuple[int, int]:
        s = self.span(z)
        w = math.ceil((self.extent[2] - self.extent[0]) / s - 1e-9)
        h = math.ceil((self.extent[3] - self.extent[1]) / s - 1e-9)
        return max(w, 1), max(h, 1)

    def tile_extent(self, z: int, x: int, y: int) -> gws.Extent:
        s = self.span(z)
        x0 = self.extent[0] + x * s
        if self.origin == 'nw':
            y1 = self.extent[3] - y * s
            y0 = y1 - s
        else:
            y0 = self.extent[1] + y * s
            y1 = y0 + s
        return x0, y0, x0 + s, y1


class Service:
    type = ''
    tmsOrigin = 'nw'
    restful = False

    def __init__(self, cfg: dict, scn: scene.Scene):
        self.cfg = cfg
        self.uid = cfg['uid']
        self.scene = scn
        self.crs = gws.lib.crs.require(cfg['crs'])
        self.layer = cfg.get('layer') or 'map'
        self.style = cfg.get('style') or 'default'
        self.version = str(cfg.get('version') or '')
        self.overlay = cfg.get('overlay', True)
        self.overlayTile = cfg.get('overlayTile', False)

        self.tms = Tms(cfg, self.crs, self.tmsOrigin) if cfg.get('resolutions') else None
        self.extent = tuple(scn.crs.transform_extent(scn.extent, self.crs))

        self.wgs = _bbox(self.crs.transform_extent(self.extent, gws.lib.crs.require(4326)))
        self.bbox = _bbox(self.axis_extent(self.extent))

    def handle(self, rest, query, base):
        raise NotImplementedError

    def tile(self, z, x, y):
        ext = self.tms.tile_extent(z, x, y)
        size = (self.tms.tileSize, self.tms.tileSize)
        label = f'{self.type} z={z} x={x} y={y}' if self.overlayTile else ''
        return self.image(ext, size, label)

    def image(self, extent, size, label=''):
        extent = tuple(extent)
        size = tuple(size)
        key = (self.scene.key, self.crs.srid, extent, size, self.overlay, label)

        with _cache_lock:
            blob = _cache.get(key)
        if blob is not None:
            return 'image', blob

        img = self.scene.render(self.crs, extent, size)
        if self.overlay:
            draw_overlay(img, '\n'.join(p for p in [
                label,
                f'{self.crs.epsg} {size[0]}x{size[1]}',
                ' '.join(f'{c:.{self.crs.coordinatePrecision}f}' for c in extent),
            ] if p))

        blob = img.to_bytes('image/png')

        with _cache_lock:
            _cache[key] = blob
        return 'image', blob

    def xml(self, func, base):
        tpl = gws.u.read_file(os.path.dirname(__file__) + '/caps.cx')
        tpl += f'\n@{func}'
        txt = gws.lib.vendor.jump.render(tpl, {'self': self, 'base': base})
        return 'xml', txt.strip()

    def axis_extent(self, ext):
        if self.crs.isYX:
            return ext[1], ext[0], ext[3], ext[2]
        return ext


class Xyz(Service):
    type = 'xyz'
    tmsOrigin = 'nw'

    def handle(self, rest, query, base):
        z, x, y = rest[0], rest[1], rest[2].split('.')[0]
        return self.tile(int(z), int(x), int(y))


class Tile(Service):
    type = 'tms'
    tmsOrigin = 'sw'

    def handle(self, rest, query, base):
        if rest == ['1.0.0']:
            return self.xml('TMS_SERVICE', base)
        if len(rest) == 2 and rest[0] == '1.0.0':
            return self.xml('TMS_MAP', base)
        z, x, y = rest[2], rest[3], rest[4].split('.')[0]
        return self.tile(int(z), int(x), int(y))


class Wmts(Service):
    type = 'wmts'
    tmsOrigin = 'nw'
    restful = False

    def handle(self, rest, query, base):
        if query.get('request', '').lower() == 'getcapabilities':
            return self.xml('WMTS', base)
        z = int(query['tilematrix'].split(':')[-1])
        return self.tile(z, int(query['tilecol']), int(query['tilerow']))


class WmtsRest(Service):
    type = 'wmts_rest'
    tmsOrigin = 'nw'
    restful = True

    def handle(self, rest, query, base):
        if len(rest) < 6:
            return self.xml('WMTS', base)
        z, y, x = rest[3], rest[4], rest[5].split('.')[0]
        return self.tile(int(z), int(x), int(y))


class Wms(Service):
    type = 'wms'

    def handle(self, rest, query, base):
        version = query.get('version') or self.version

        if query.get('request', '').lower() == 'getcapabilities':
            return self.xml('WMS_111' if version == '1.1.1' else 'WMS_130', base)

        bbox = [float(s) for s in query['bbox'].split(',')]
        if version == '1.3.0' and self.crs.isYX:
            bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
        size = (int(query['width']), int(query['height']))
        return self.image(bbox, size)


def draw_overlay(img: gws.Image, text: str):
    img.add_box((60, 60, 60, 160))
    draw = gws.lib.image.get_draw(img)
    font = gws.lib.image.get_font(11, scene.FONT_PATH)
    draw.multiline_text(
        (5, 4), text,
        font=font,
        fill=(0, 0, 0, 255),
        stroke_width=2,
        stroke_fill=(255, 255, 255, 220),
    )


def _bbox(ext) -> gws.Data:
    return gws.Data(minx=ext[0], miny=ext[1], maxx=ext[2], maxy=ext[3])


TYPES = {
    'xyz': Xyz,
    'tms': Tile,
    'wmts': Wmts,
    'wmts_rest': WmtsRest,
    'wms': Wms,
}


def create(cfg: dict, scn: scene.Scene) -> Service:
    return TYPES[cfg['type']](cfg, scn)
