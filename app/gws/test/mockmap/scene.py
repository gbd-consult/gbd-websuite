"""Mockmap scene: synthetic geometry rendered by MapServer."""

import math
import threading

import gws
import gws.lib.crs
import gws.lib.image
import gws.lib.mapserver
import gws.lib.vendor.jump

FONT_NAME = 'sans'
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
SYMBOLSET_PATH = gws.c.APP_DIR + '/gws/lib/mapserver/symbolset.sym'

DENSIFY = 10
CIRCLE_SEGMENTS = 72

SCENE_CRS = 4326

KEYS = [
    'extent',
    'step',
    'graticule',
    'checker',
    'shapes',
    'labels',
    'labelSize',
    'labelPad',
    'labelPartials',
    'labelForce',
    'background',
]

MAPFILE = '''
MAP
    NAME "mockmap"
    SIZE 256 256
    EXTENT {self.extentString}
    IMAGETYPE "png"
    FONTSET "{self.fontset}"
    SYMBOLSET "{self.symbolset}"
    @for la in self.layers
        LAYER
            NAME "{la.name}"
            TYPE {la.type}
            STATUS ON
            @for f in la.features
                FEATURE 
                    POINTS {f.points} END
                    @if f.text 
                        TEXT "{f.text}"
                    @end 
                END
            @end
            CLASS
                {la.style}
                @if la.label
                    LABEL
                        FONT "{la.label.font}"
                        TYPE TRUETYPE
                        SIZE {la.label.size}
                        COLOR "#000000"
                        OUTLINECOLOR "#ffffff"
                        OUTLINEWIDTH 2
                        POSITION UR
                        OFFSET 5 -5
                        PARTIALS {la.label.partials}
                        FORCE {la.label.force}
                    END
                @end
            END
        END
    @end
END
'''

_fontset_path = ''
_fontset_lock = threading.Lock()


def fontset() -> str:
    global _fontset_path
    with _fontset_lock:
        if not _fontset_path:
            p = gws.c.EPHEMERAL_DIR + '/mockmap_fonts.txt'
            gws.u.write_file(p, f'{FONT_NAME} {FONT_PATH}\n')
            _fontset_path = p
    return _fontset_path


class Scene:
    def __init__(self, cfg: dict):
        self.cfg = {k: cfg[k] for k in KEYS}

        self.crs = gws.lib.crs.require(SCENE_CRS)
        self.extent = tuple(self.cfg['extent'])
        self.step = float(self.cfg['step'])

        self.fontset = fontset()
        self.symbolset = SYMBOLSET_PATH
        self.extentString = _nums(self.extent)
        self.layers = self._layers()

        self.key = repr(sorted((k, repr(v)) for k, v in self.cfg.items()))
        self.mapfile = gws.lib.vendor.jump.render(MAPFILE, {'self': self})
        self.local = threading.local()

    def render(self, crs: gws.Crs, extent: gws.Extent, size: gws.Size) -> gws.Image:
        # MapServer forces square pixels, silently widening the extent when the
        # bbox aspect does not match the requested size. Draw with square pixels
        # over the exact extent, then scale to the requested size.

        res = min((extent[2] - extent[0]) / size[0], (extent[3] - extent[1]) / size[1])
        draw_size = (
            max(1, round((extent[2] - extent[0]) / res)),
            max(1, round((extent[3] - extent[1]) / res)),
        )

        img = self._map().draw(gws.Bounds(crs=crs, extent=extent), draw_size)
        if draw_size != tuple(size):
            img.resize(size)

        bg = self.cfg['background']
        if bg:
            base = gws.lib.image.from_size(size, bg)
            return base.compose(img)

        return img

    def _map(self) -> gws.lib.mapserver.Map:
        m = getattr(self.local, 'map', None)
        if m is None:
            m = gws.lib.mapserver.new_map(self.mapfile)
            for n in range(m.mapObj.numlayers):
                m.mapObj.getLayer(n).setProjection(self.crs.epsg)
            self.local.map = m
        return m

    def _layers(self) -> list[gws.Data]:
        out = []

        if self.cfg['checker']:
            out.append(gws.Data(
                name='checker',
                type='POLYGON',
                style='STYLE COLOR "#e8e8f080" OUTLINECOLOR "#c8c8d2" WIDTH 1 END',
                label=None,
                features=self._checker_features(),
            ))

        if self.cfg['graticule']:
            out.append(gws.Data(
                name='graticule',
                type='LINE',
                style='STYLE COLOR "#be2828" WIDTH 1 END',
                label=None,
                features=self._graticule_features(),
            ))

        if self.cfg['shapes']:
            out.append(gws.Data(
                name='shapes',
                type='LINE',
                style='STYLE COLOR "#1464c8" WIDTH 2 END',
                label=None,
                features=self._shape_features(),
            ))

        if self.cfg['labels']:
            out.append(gws.Data(
                name='labels',
                type='POINT',
                style='STYLE SYMBOL "circle" SIZE 5 COLOR "#141414" END',
                label=gws.Data(
                    font=FONT_NAME,
                    size=self.cfg['labelSize'],
                    partials=_bool(self.cfg['labelPartials']),
                    force=_bool(self.cfg['labelForce']),
                ),
                features=self._label_features(),
            ))

        return out

    def _graticule_features(self) -> list[gws.Data]:
        x0, y0, x1, y1 = self.extent
        d = self.step / DENSIFY
        out = []

        for x in _steps(x0, x1, self.step):
            out.append(_feature([(x, y) for y in _steps(y0, y1, d)]))
        for y in _steps(y0, y1, self.step):
            out.append(_feature([(x, y) for x in _steps(x0, x1, d)]))

        return out

    def _checker_features(self) -> list[gws.Data]:
        x0, y0, x1, y1 = self.extent
        s = self.step
        d = s / DENSIFY
        out = []

        for i, x in enumerate(_steps(x0, x1 - s / 2, s)):
            for j, y in enumerate(_steps(y0, y1 - s / 2, s)):
                if (i + j) % 2:
                    continue
                a = [(x, py) for py in _steps(y, y + s, d)]
                b = [(px, y + s) for px in _steps(x, x + s, d)]
                c = [(x + s, py) for py in _steps(y + s, y, -d)]
                e = [(px, y) for px in _steps(x + s, x, -d)]
                out.append(_feature(a + b + c + e))

        return out

    def _shape_features(self) -> list[gws.Data]:
        x0, y0, x1, y1 = self.extent
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        s = self.step
        out = []

        r = 3 * s
        pts = []
        for n in range(CIRCLE_SEGMENTS + 1):
            a = 2 * math.pi * n / CIRCLE_SEGMENTS
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        out.append(_feature(pts))

        d = s / DENSIFY
        sq = (
            [(cx - 2 * s, py) for py in _steps(cy - 2 * s, cy + 2 * s, d)]
            + [(px, cy + 2 * s) for px in _steps(cx - 2 * s, cx + 2 * s, d)]
            + [(cx + 2 * s, py) for py in _steps(cy + 2 * s, cy - 2 * s, -d)]
            + [(px, cy - 2 * s) for px in _steps(cx + 2 * s, cx - 2 * s, -d)]
        )
        out.append(_feature(sq))

        out.append(_feature([(px, cy) for px in _steps(cx - 3 * s, cx + 3 * s, d)]))
        out.append(_feature([(cx, py) for py in _steps(cy - 3 * s, cy + 3 * s, d)]))

        return out

    def _label_features(self) -> list[gws.Data]:
        x0, y0, x1, y1 = self.extent
        pad = '-' * int(self.cfg['labelPad'])
        out = []

        for x in _steps(x0, x1, self.step):
            for y in _steps(y0, y1, self.step):
                out.append(_feature([(x, y)], f'{_num(x)} {_num(y)}{pad}'))

        return out


def _feature(points: list[tuple], text: str = '') -> gws.Data:
    return gws.Data(
        points=' '.join(f'{_num(x)} {_num(y)}' for x, y in points),
        text=text,
    )


def _steps(a: float, b: float, step: float) -> list[float]:
    n = int(math.floor((b - a) / step + 1e-9))
    return [a + i * step for i in range(n + 1)]


def _num(x: float) -> str:
    return f'{x:.6f}'.rstrip('0').rstrip('.') or '0'


def _nums(xs) -> str:
    return ' '.join(_num(x) for x in xs)


def _bool(x) -> str:
    return 'TRUE' if x else 'FALSE'
