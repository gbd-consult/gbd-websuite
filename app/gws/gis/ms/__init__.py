"""MapServer support.

This module dynamically creates and renders MapServer maps.

To render a map, create a map object with `new_map`, add layers to it using ``add_`` methods
and invoke ``draw``.

Reference: MapServer documentation (https://mapserver.org/documentation.html)

Example usage::

    import gws.gis.ms as ms

    # create a new map
    map = ms.new_map()

    # add a raster layer from an image file
    map.add_layer(
        ms.LayerOptions(
            type=ms.LayerType.raster,
            path='/path/to/image.tif',
        )
    )

    # add a layer using a configuration string
    map.add_layer_from_config('''
        LAYER
            TYPE LINE
            STATUS ON
            FEATURE
                POINTS
                    751539 6669003
                    751539 6672326
                    755559 6672326
                END
            END
            CLASS
                STYLE
                    COLOR 0 255 0
                    WIDTH 5
                END
            END
        END
    ''')

    # draw the map into an Image object
    img = map.draw(
        bounds=gws.Bounds(
            extent=[738040, 6653804, 765743, 6683686],
            crs=gws.lib.crs.WEBMERCATOR,
        ),
        size=(800, 600),
    )

    # save the image to a file
    img.to_path('/path/to/output.png')


"""

from typing import Optional
import mapscript

import gws
import gws.lib.image


def version() -> str:
    """Returns the MapServer version string."""

    return mapscript.msGetVersion()


class Error(gws.Error):
    pass


class LayerType(gws.Enum):
    """MapServer layer type."""

    point = 'point'
    line = 'line'
    polygon = 'polygon'
    raster = 'raster'


_LAYER_TYPE_TO_MS = {
    LayerType.point: mapscript.MS_LAYER_POINT,
    LayerType.line: mapscript.MS_LAYER_LINE,
    LayerType.polygon: mapscript.MS_LAYER_POLYGON,
    LayerType.raster: mapscript.MS_LAYER_RASTER,
}


class LayerOptions(gws.Data):
    """Options for a mapserver layer."""

    type: LayerType
    """Layer type."""
    path: str
    """Path to the image file."""
    tileIndex: str
    """Path to the tile index SHP file"""
    crs: gws.Crs
    """Layer CRS."""
    connectionType: str
    """Type of connection (e.g., 'postgres')."""
    connectionString: str
    """Connection string for the data source."""
    dataString: str
    """Layer DATA option."""
    style: gws.StyleValues
    """Style for the layer."""
    processing: list[str]
    """Processing options for the layer."""
    transparentColor: str
    """Color to treat as transparent in the layer (OFFSITE)."""
    sldPath: str
    """Path to SLD file for styling the layer."""
    sldName: str
    """Name of an SLD NamedLayer to apply."""


def new_map(config: str = '') -> 'Map':
    """Creates a new Map instance from a Mapfile string."""

    return Map(config)


class Map:
    """MapServer map object wrapper."""

    mapObj: mapscript.mapObj

    def __init__(self, config: str = ''):
        if config:
            tmp = gws.c.EPHEMERAL_DIR + '/mapse_' + gws.u.random_string(16) + '.map'
            gws.u.write_file(tmp, config)
            self.mapObj = mapscript.mapObj(tmp)
        else:
            self.mapObj = mapscript.mapObj()

        self.mapObj.setConfigOption('MS_ERRORFILE', 'stderr')
        # self.mapObj.debug = mapscript.MS_DEBUGLEVEL_DEVDEBUG
        self.mapObj.debug = mapscript.MS_DEBUGLEVEL_ERRORSONLY

    def copy(self) -> 'Map':
        """Creates a copy of the current map object."""

        c = Map()
        c.mapObj = self.mapObj.clone()
        return c

    def add_layer_from_config(self, config: str) -> mapscript.layerObj:
        """Adds a layer to the map using a configuration string."""

        try:
            lo = mapscript.layerObj(self.mapObj)
            lo.updateFromString(config)
            return lo
        except mapscript.MapServerError as exc:
            raise Error(f'ms: add error:: {exc}') from exc

    def add_layer(self, opts: LayerOptions) -> mapscript.layerObj:
        """Adds a layer to the map."""

        try:
            lo = self._make_layer(opts)
            return lo
        except mapscript.MapServerError as exc:
            raise Error(f'ms: add error:: {exc}') from exc

    def _make_layer(self, opts: LayerOptions) -> mapscript.layerObj:
        lo = mapscript.layerObj(self.mapObj)
        lc = self.mapObj.numlayers
        lo.name = f'_gws_{lc}'
        lo.status = mapscript.MS_ON

        if not opts.crs:
            raise Error('missing layer CRS')
        lo.setProjection(opts.crs.epsg)

        if opts.type:
            lo.type = _LAYER_TYPE_TO_MS[opts.type]
        if opts.path:
            lo.data = opts.path
        if opts.tileIndex:
            lo.tileindex = opts.tileIndex
        if opts.processing:
            for p in opts.processing:
                lo.addProcessing(p)
        if opts.transparentColor:
            co = mapscript.colorObj()
            co.setHex(opts.transparentColor)
            lo.offsite = co
        if opts.connectionType:
            if opts.connectionType == 'postgres':
                lo.setConnectionType(mapscript.MS_POSTGIS, '')
            else:
                raise Error(f'unsupported connectionType {opts.connectionType!r}')
        if opts.connectionString:
            lo.connection = opts.connectionString
        if opts.dataString:
            lo.data = opts.dataString
        if opts.sldPath:
            lo.applySLD(gws.u.read_file(opts.sldPath), opts.sldName)

        # @TODO: support style values
        if opts.style:
            cls = mapscript.classObj(lo)

            if opts.style.with_geometry == 'all':
                style_obj = self._create_style_obj(opts.style)
                cls.insertStyle(style_obj)

            if opts.style.with_label == 'all':
                label_obj = self._create_label_obj(opts.style)
                cls.addLabel(label_obj)
                lo.labelitem = 'label'

            if opts.style.marker or opts.style.icon:
                if opts.style.marker:
                    self.mapObj.setSymbolSet('/gws-app/gws/gis/ms/symbolset.sym')
                    so = self.style_symbol(opts.style)
                    cls.insertStyle(so)

                if opts.style.icon:
                    symbol = mapscript.symbolObj("icon", opts.style.icon)
                    symbol.type = mapscript.MS_SYMBOL_PIXMAP
                    lo.map.symbolset.appendSymbol(symbol)
                    so = mapscript.styleObj()
                    so.setSymbolByName(lo.map, "icon")
                    so.size = 100
                    cls.insertStyle(so)
        return lo

    def draw(self, bounds: gws.Bounds, size: gws.Size) -> gws.Image:
        """Renders the map within the given bounds and size.

        Args:
            bounds: The spatial extent to render.
            size: The output image size.

        Returns:
            The rendered map image.
        """

        # @TODO: options for image format, transparency, etc.

        try:
            gws.debug.time_start(f'mapserver.draw {bounds=} {size=}')

            self.mapObj.setOutputFormat(mapscript.outputFormatObj('AGG/PNG'))
            self.mapObj.outputformat.transparent = mapscript.MS_TRUE

            self.mapObj.setExtent(*bounds.extent)
            self.mapObj.setSize(*size)
            self.mapObj.setProjection(bounds.crs.epsg)

            res = self.mapObj.draw()
            img = gws.lib.image.from_bytes(res.getBytes())

            gws.debug.time_end()

            return img

        except mapscript.MapServerError as exc:
            raise Error(f'ms: draw error: {exc}') from exc

    def to_string(self) -> str:
        """Converts the map object to a configuration string."""

        try:
            return self.mapObj.convertToString()
        except mapscript.MapServerError as exc:
            raise Error(f'ms: convert error: {exc}') from exc

    def _create_style_obj(self, style: gws.StyleValues) -> mapscript.styleObj:
        so = mapscript.styleObj()
        if style.fill:
            so.color.setRGB(*_css_color_to_rgb(style.fill))
        if style.stroke:
            so.outlinecolor.setRGB(*_css_color_to_rgb(style.stroke))
            so.outlinewidth = max(0.1 * style.stroke_width, 1)
        if style.stroke_dasharray:
            so.pattern_set(style.stroke_dasharray)
        if style.stroke_dashoffset:
            so.gap = style.stroke_dashoffset
        if style.stroke_linecap:
            so.linecap = _const_mapping.get(style.stroke_linecap.lower())
        if style.stroke_linejoin:
            so.linejoin = _const_mapping.get(style.stroke_linejoin.lower())
        if style.stroke_miterlimit:
            so.linejoinmaxsize = style.stroke_miterlimit
        if style.stroke_width:
            so.width = style.stroke_width
        if style.offset_x:
            so.offsetx = style.offset_x
        if style.offset_y:
            so.offsety = style.offset_y
        return so

    def _create_label_obj(self, style: gws.StyleValues) -> mapscript.labelObj:
        lo = mapscript.labelObj()
        so = mapscript.styleObj()
        lo.force = mapscript.MS_TRUE

        if style.label_align:
            lo.align = _const_mapping.get(style.label_align)
        if style.label_background:
            so.setGeomTransform('labelpoly')
            so.color.setRGB(*_css_color_to_rgb(style.label_background))
        if style.label_fill:
            lo.color.setRGB(*_css_color_to_rgb(style.label_fill))
        if style.label_font_family:
            lo.font = style.label_font_family  # + '-' + style.label_font_style + '-' + style.label_font_weight
        if style.label_font_size:
            lo.size = style.label_font_size
        if style.label_max_scale:
            lo.maxscaledenom = style.label_max_scale
        if style.label_min_scale:
            lo.minscaledenom = style.label_min_scale
        if style.label_offset_x:
            lo.offsetx = style.label_offset_x
        if style.label_offset_y:
            lo.offsety = style.label_offset_y
        if style.label_padding:
            lo.buffer = max(style.label_padding)
        if style.label_placement:
            lo.position = _const_mapping.get(style.label_placement)
        if style.label_stroke:
            lo.outlinecolor.setRGB(*_css_color_to_rgb(style.label_stroke))
        if style.label_stroke_dasharray:
            so.pattern_set(style.label_stroke_dasharray)
        if style.label_stroke_linecap:
            so.linecap = _const_mapping.get(style.label_stroke_linecap.lower())
        if style.label_stroke_linejoin:
            so.linejoin = _const_mapping.get(style.label_stroke_linejoin.lower())
        if style.label_stroke_miterlimit:
            so.linejoinmaxsize = style.label_stroke_miterlimit
        if style.label_stroke_width:
            lo.outlinewidth = style.label_stroke_width
        lo.insertStyle(so)
        return lo

    def style_symbol(self, style: gws.StyleValues) -> mapscript.styleObj:
        mo = self.mapObj
        so = mapscript.styleObj()
        so.setSymbolByName(mo, style.marker)

        if style.marker_fill:
            so.color.setRGB(*_css_color_to_rgb(style.marker_fill))
        if style.marker_size:
            so.size = style.marker_size
        if style.marker_stroke:
            so.outlinecolor.setRGB(*_css_color_to_rgb(style.marker_stroke))
        if style.marker_stroke_dasharray:
            so.pattern_set(style.marker_stroke_dasharray)
        if style.marker_stroke_dashoffset:
            so.gap = style.marker_stroke_dashoffset
        if style.marker_stroke_linecap:
            so.linecap = _const_mapping.get(style.marker_stroke_linecap.lower())
        if style.marker_stroke_linejoin:
            so.linejoin = _const_mapping.get(style.marker_stroke_linejoin.lower())
        if style.marker_stroke_miterlimit:
            so.linejoinmaxsize = style.marker_stroke_miterlimit
        if style.marker_stroke_width:
            so.outlinewidth = style.marker_stroke_width
        return so

def _css_color_to_rgb(color_name: str) -> tuple[int, int, int]:
    try:
        return _CSS_COLOR_NAMES[color_name.lower()]
    except KeyError:
        raise ValueError(f"Unbekannter CSS-Farbenname: '{color_name}'")


def _color_to_str(color: str) -> str:
    if isinstance(color, str):
        r, g, b = _css_color_to_rgb(color)
    return f"{r} {g} {b}"

_CSS_COLOR_NAMES = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'lime': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'aqua': (0, 255, 255),
    'magenta': (255, 0, 255),
    'fuchsia': (255, 0, 255),
    'gray': (128, 128, 128),
    'grey': (128, 128, 128),
    'maroon': (128, 0, 0),
    'olive': (128, 128, 0),
    'green': (0, 128, 0),
    'purple': (128, 0, 128),
    'teal': (0, 128, 128),
    'navy': (0, 0, 128),
}

_const_mapping = {
    'butt': mapscript.MS_CJC_BUTT,
    'round': mapscript.MS_CJC_ROUND,
    'square': mapscript.MS_CJC_SQUARE,
    'bevel': mapscript.MS_CJC_BEVEL,
    'miter': mapscript.MS_CJC_MITER,
    'left': mapscript.MS_ALIGN_LEFT,
    'center': mapscript.MS_ALIGN_CENTER,
    'right': mapscript.MS_ALIGN_RIGHT,
    'start': mapscript.MS_CL,
    'middle': mapscript.MS_CC,
    'end': mapscript.MS_CR,
}
